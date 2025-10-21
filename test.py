"""
Test/Inference Script for Enhanced LeSICiN
"""
import torch
import json
import sys
sys.path.append('modules')

from train import SimpleHAN, LegalDataset, evaluate
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config and vocabularies
    config = checkpoint['config']
    word_to_idx = checkpoint['word_to_idx']
    label_to_idx = checkpoint['label_to_idx']
    
    # Rebuild model
    model = SimpleHAN(
        vocab_size=len(word_to_idx),
        hidden_size=config['hidden_size'],
        num_labels=len(label_to_idx),
        orwell_dim=3
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (Best F1: {checkpoint['best_f1']:.2f}%)")
    
    return model, config, word_to_idx, label_to_idx


def predict_single(text, model, word_to_idx, label_to_idx, device, 
                   use_trie=True, use_orwell=True, threshold=0.65):
    """Predict IPC sections for a single case description"""
    from modules.trie_annotator import TrieAnnotator
    from modules.orwell_simplifier import OrwellSimplifier
    
    # Initialize preprocessors
    if use_trie:
        trie = TrieAnnotator()
        text = trie.annotate_text(text)
    
    if use_orwell:
        orwell = OrwellSimplifier()
        orwell_features = orwell.extract_features(text)
    else:
        orwell_features = [0.0, 0.0, 0.0]
    
    # Encode text
    max_segments = 50
    max_words = 100
    encoded = []
    
    if isinstance(text, str):
        text = [text]
    
    for sent in text[:max_segments]:
        words = sent.lower().split()[:max_words]
        word_ids = [word_to_idx.get(w, 1) for w in words]
        word_ids += [0] * (max_words - len(word_ids))
        encoded.append(word_ids)
    
    while len(encoded) < max_segments:
        encoded.append([0] * max_words)
    
    # Convert to tensors
    text_tensor = torch.LongTensor(encoded).unsqueeze(0).to(device)
    orwell_tensor = torch.FloatTensor(orwell_features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(text_tensor, orwell_tensor)
        probs = torch.sigmoid(logits)[0]
    
    # Get predictions above threshold
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    predictions = []
    
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            predictions.append({
                'section': idx_to_label[idx],
                'probability': prob.item()
            })
    
    # Sort by probability
    predictions.sort(key=lambda x: -x['probability'])
    
    return predictions


def test_on_dataset():
    """Test on the test dataset"""
    print("="*80)
    print("Testing Enhanced LeSICiN")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model, config, word_to_idx, label_to_idx = load_model(
        'output/best_model.pt',
        device
    )
    
    # Load test data
    print("\nLoading test data...")
    with open('configs/data_paths.json', 'r') as f:
        data_paths = json.load(f)
    
    test_dataset = LegalDataset(
        data_paths['test_src'],
        word_to_idx=word_to_idx,
        label_to_idx=label_to_idx,
        max_segments=config['max_segments'],
        max_words=config['max_segment_size'],
        use_trie=config['use_trie'],
        use_orwell=config['use_orwell']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test_bs'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test set size: {len(test_dataset):,} instances")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, threshold=config['pthresh'])
    
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    print(f"\nMacro Metrics:")
    print(f"  Precision: {test_metrics['macro']['precision']:.2f}%")
    print(f"  Recall:    {test_metrics['macro']['recall']:.2f}%")
    print(f"  F1:        {test_metrics['macro']['f1']:.2f}%")
    
    print(f"\nMicro Metrics:")
    print(f"  Precision: {test_metrics['micro']['precision']:.2f}%")
    print(f"  Recall:    {test_metrics['micro']['recall']:.2f}%")
    print(f"  F1:        {test_metrics['micro']['f1']:.2f}%")
    
    print(f"\nJaccard Index: {test_metrics['jaccard']:.2f}%")
    
    # Save results
    with open(data_paths['test_metrics_dump'], 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n✓ Results saved to: {data_paths['test_metrics_dump']}")
    
    print("="*80)


def interactive_demo():
    """Interactive demo for testing on custom inputs"""
    print("="*80)
    print("Interactive Demo - Enhanced LeSICiN")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, config, word_to_idx, label_to_idx = load_model(
        'output/best_model.pt',
        device
    )
    
    print("\nEnter case descriptions to predict IPC sections.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-" * 80)
        text = input("\nCase description: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        # Predict
        predictions = predict_single(
            text, model, word_to_idx, label_to_idx, device,
            use_trie=config['use_trie'],
            use_orwell=config['use_orwell'],
            threshold=config['pthresh']
        )
        
        # Display results
        print("\nPredicted IPC Sections:")
        if predictions:
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. IPC {pred['section']} "
                      f"(confidence: {pred['probability']*100:.1f}%)")
        else:
            print("  No sections predicted above threshold.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced LeSICiN')
    parser.add_argument('--mode', choices=['test', 'demo'], default='test',
                       help='test: evaluate on test set, demo: interactive')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_on_dataset()
    else:
        interactive_demo()