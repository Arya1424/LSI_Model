import torch
import random
import numpy as np
import json
import pickle as pkl
from tqdm import tqdm
from functools import partial
import argparse
import sys
import multiprocessing
from multiprocessing import freeze_support 
import os 

# Add modules path and import new features
sys.path.append('./modules') 
from kdtree_retrieval import KDTreeRetriever

# Core LeSICiN imports
from model.model import LeSICiN 
from data_helper import LSIDataset, collate_func
from helper import generate_vocabs, generate_graph, generate_label_weights, train_dev_pass, MultiLabelMetrics


torch.autograd.set_detect_anomaly(True)


# --- CONFIG LOADING ---
with open("configs/data_path.json") as fr:
    dc = json.load(fr)
with open("configs/hyperparams.json") as fr:
    hc = json.load(fr)

SEED = hc['seed']

random.seed(SEED)
np.random.seed(SEED)

# Define Device 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# NEW: Feature flags loaded from hyperparams
use_trie = hc.get('use_trie', False)
use_orwell = hc.get('use_orwell', False)
use_kdtree = hc.get('use_kdtree', False)
orwell_dim = hc.get('orwell_dim', 3)
k_neighbors = hc.get('k_neighbors', 10)

# --- ROBUST SENT2VEC LOADING BLOCK ---
# --- ROBUST SENT2VEC LOADING BLOCK ---
sent2vec_model = None
if dc['s2v_path'] is not None and dc['s2v_path'] != "null":
    try:
        import sent2vec
        print(f"\n=== Loading Sent2Vec Model ===")
        print(f"Model path: {dc['s2v_path']}")
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(dc['s2v_path'])
        
        # Test the model
        test_embed = sent2vec_model.embed_sentence("test sentence")
        print(f"✓ Sent2Vec model loaded successfully")
        print(f"✓ Embedding dimension: {test_embed.shape}")
        print(f"✓ Test embedding sum: {test_embed.sum()}")
        print("=" * 50)
    except ImportError:
        print("!! WARNING: sent2vec library not installed.")
        print("!! Install with: pip install sent2vec")
        sent2vec_model = None
    except FileNotFoundError:
        print(f"!! WARNING: Sent2Vec model file not found at {dc['s2v_path']}")
        sent2vec_model = None
    except Exception as e:
        print(f"!! WARNING: Could not load Sent2Vec model: {e}")
        sent2vec_model = None
else:
    print("!! WARNING: No sent2vec path configured (s2v_path is null)")
    sent2vec_model = None
# ------------------------------------


def load_or_preprocess_dataset(data_src, cache_path, sent2vec_model, use_trie, use_orwell):
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} (Skipping preprocessing)...")
        try:
            return LSIDataset.load_data(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}. Re-running preprocessing.")
    
    print(f"Cache not found or failed. Starting full preprocessing for {data_src}...")
    dataset = LSIDataset(jsonl_file=data_src)
    dataset.preprocess(use_trie=use_trie, use_orwell=use_orwell)
    
    # CHANGED: Don't vectorize if no sent2vec - keep as text tokens
    if sent2vec_model is not None:
        dataset.sent_vectorize(sent2vec_model)
    else:
        print("⚠ Training with word embeddings (no sent2vec)")
        # Don't call sent_vectorize - keep text as tokens
    
    dataset.save_data(cache_path)
    return dataset

# --- Main Execution Block ---
def main_execution():
    torch.manual_seed(SEED)
    
    print("\nPreparing PyTorch environment")
    print("==================================================")
    
    # Device info
    device_info = str(DEVICE.type).upper()
    if DEVICE.type == 'cuda' and torch.cuda.device_count() > 0:
        device_name = torch.cuda.get_device_name(0)
        print(f"Using Device: {device_name} (CUDA)")
    else:
        print(f"Using Device: CPU")
    print("--------------------------------------------------")
    
    print("\nPreparing Datasets")
    print("==================================================")
    
    # Always load sections
    sec_dataset = load_or_preprocess_dataset(
        dc['sec_src'], dc['sec_cache'], sent2vec_model, use_trie, use_orwell
    )
    
    # Initialize variables to avoid UnboundLocalError
    train_dataset = None
    dev_dataset = None
    test_dataset = None
    infer_dataset = None

    # Conditionally load other datasets
    if hc['do_train_dev']:
        train_dataset = load_or_preprocess_dataset(
            dc['train_src'], dc['train_cache'], sent2vec_model, use_trie, use_orwell
        )
        dev_dataset = load_or_preprocess_dataset(
            dc['dev_src'], dc['dev_cache'], sent2vec_model, use_trie, use_orwell
        )

    if hc['do_test']:
        test_dataset = load_or_preprocess_dataset(
            dc['test_src'], dc['test_cache'], sent2vec_model, use_trie, use_orwell
        )

    if hc['do_infer']:
        infer_dataset = load_or_preprocess_dataset(
            dc['infer_src'], dc['infer_cache'], sent2vec_model, use_trie, use_orwell
        )
    
    # === DATA VERIFICATION ===
    print("\n=== DATA VERIFICATION ===")
    print(f"Sections: {len(sec_dataset)} items")
    
    if train_dataset is not None:
        print(f"Train: {len(train_dataset)} items")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample train item:")
            print(f"  ID: {sample['id']}")
            print(f"  Text type: {type(sample['text'])}")
            if hasattr(sample['text'], 'shape'):
                print(f"  Text shape: {sample['text'].shape}")
            if hasattr(sample['text'], '__len__') and len(sample['text']) > 0:
                print(f"  Number of sentences: {len(sample['text'])}")
                first_sent = str(sample['text'][0])
                print(f"  First sentence: {first_sent[:100]}{'...' if len(first_sent) > 100 else ''}")
            else:
                print(f"  WARNING: Text is EMPTY!")
            
            if 'labels' in sample:
                labels = sample['labels']
                if isinstance(labels, np.ndarray):
                    print(f"  Labels: {labels} (count: {len(labels)})")
                else:
                    print(f"  Labels: {labels}")
            else:
                print(f"  WARNING: No labels found!")
            
            if 'orwell_features' in sample:
                print(f"  Orwell features: {sample['orwell_features']}")
    
    if dev_dataset is not None:
        print(f"Dev: {len(dev_dataset)} items")
    
    if test_dataset is not None:
        print(f"Test: {len(test_dataset)} items")
    
    if infer_dataset is not None:
        print(f"Infer: {len(infer_dataset)} items")
    
    print("=========================\n")

    # Continue with the rest of your code...
    print("\nGathering other data")
    print("==================================================")
    
    # Make sure train_dataset is not None before using it
    if train_dataset is None:
        print("ERROR: train_dataset is None! Cannot proceed with training.")
        return
    
    vocab, label_vocab = generate_vocabs(train_dataset, sec_dataset, limit=hc['vocab_limit'], thresh=hc['vocab_thresh'])
    
    # ... rest of your code
    with open(dc['type_map']) as fr:
        type_map = json.load(fr)
    with open(dc['label_tree']) as fr:
        label_tree = json.load(fr)
    with open(dc['citation_network']) as fr:
        citation_net = json.load(fr)
    with open(dc['schemas']) as fr:
        schemas = json.load(fr)
    for sch in schemas.values():
        for path in sch:
            for i, edge in enumerate(path):
                path[i] = tuple(path[i])

    node_vocab, edge_vocab, edge_indices, adjacency = generate_graph(label_vocab, type_map, label_tree, citation_net)
    
    sec_weights = generate_label_weights(
        train_dataset, 
        label_vocab, 
        dev=str(DEVICE), 
        scheme=hc['weight_scheme'], 
        thresh=hc['tws_thresh']
    )

    L = len(label_vocab)
    N = {k: len(v) for k,v in node_vocab.items()}
    E = len(edge_vocab)

    sec_loader = torch.utils.data.DataLoader(
        sec_dataset, 
        batch_size=len(label_vocab), 
        collate_fn=partial(
            collate_func,
            vocab=vocab,  # ADD THIS
            schemas=schemas['section'], 
            type_map=type_map, 
            node_vocab=node_vocab, 
            edge_vocab=edge_vocab, 
            adjacency=adjacency, 
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size'],
            num_mpath_samples=hc['num_mpath_samples']
        ), 
        pin_memory=True, 
        num_workers=0 
)

    # FIX: Explicitly retrieve sec_batch and handle StopIteration
    try:
        sec_batch = next(iter(sec_loader))
    except StopIteration:
        print("\nFATAL ERROR: Section data loader is empty. Cannot continue training.")
        return

    if hc['do_train_dev']:
       train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=hc['train_bs'], 
        collate_fn=partial(
            collate_func,
            vocab=vocab,  # ADD THIS
            label_vocab=label_vocab, 
            schemas=schemas['fact'],  
                type_map=type_map, 
                node_vocab=node_vocab, 
                edge_vocab=edge_vocab, 
                adjacency=adjacency, 
                max_segments=hc['max_segments'],
                max_segment_size=hc['max_segment_size'], 
                num_mpath_samples=hc['num_mpath_samples']
                ), 
            pin_memory=True, 
            num_workers=0 
        )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=hc['dev_bs'], 
        collate_fn=partial(
            collate_func,
            vocab=vocab,  # ADD THIS
            label_vocab=label_vocab,  
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size']
        ), 
        pin_memory=True, 
        num_workers=0 
    )
    if hc['do_test']:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=hc['test_bs'], 
            collate_fn=partial(
                collate_func, 
                label_vocab=label_vocab,  
                max_segments=hc['max_segments'],
                max_segment_size=hc['max_segment_size']
                ), 
            pin_memory=True, 
            num_workers=0 
        )

    if hc['do_infer']:
        infer_loader = torch.utils.data.DataLoader(
            infer_dataset, 
            batch_size=hc['infer_bs'], 
            collate_fn=partial(
                collate_func, 
                label_vocab=label_vocab,  
                max_segments=hc['max_segments'],
                max_segment_size=hc['max_segment_size']
                ), 
            pin_memory=True, 
            num_workers=0 
        )

    # Around line 170 in main_execution()
    print("\nPreparing Model")
    print("==================================================")

    # Determine vocab size
    vocab_size = len(vocab) if vocab is not None else None

    lsc_model = LeSICiN(
        hc['hidden_size'], 
        L, 
        N, 
        E,
        vocab_size=vocab_size,  # Pass vocab size
        label_weights=sec_weights, 
        lambdas=hc['lambdas'], 
        thetas=hc['thetas'], 
        pthresh=hc['pthresh'], 
        drop=hc['dropout'],
        orwell_dim=orwell_dim 
    ).to(DEVICE)

    if dc['model_load'] is not None:
        lsc_model.load_state_dict(torch.load(dc['model_load'], map_location=DEVICE))

    kdtree = None
    if use_kdtree:
        kdtree = KDTreeRetriever(k_neighbors=k_neighbors)

    if hc['do_train_dev']:
        if dc['metrics_load'] is not None:
            with open(dc['metrics_dump'], 'rb') as fr:
               best_metrics = pkl.load(fr)
               best_loss = best_metrics.loss
        else:
            best_loss = float('inf')

        best_model = lsc_model.state_dict()


        optimizer = torch.optim.AdamW(lsc_model.parameters(), lr=hc['opt_lr'], weight_decay=hc['opt_wt_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=hc['sch_patience'], factor=hc['sch_factor'])
        train_mlmetrics = MultiLabelMetrics(L, dev=str(DEVICE))
        dev_mlmetrics = MultiLabelMetrics(L, dev=str(DEVICE))

        print("\nRunning Train/Dev")
        print("==================================================")
        
        # Add Header for Epoch Output
        print("\nEpoch || Train Loss | T_Macro-F1 || Dev Loss | D_Macro-P D_Macro-R D_Macro-F1")
        print("-------------------------------------------------------------------------------")
        
        for epoch in range(hc['num_epochs']):
            print(f"Starting Epoch {epoch}...")
            
            # Train pass - DON'T call calculate_metrics again
            train_output = train_dev_pass(lsc_model, optimizer, train_loader, sec_batch, metrics=train_mlmetrics, train=True, pred_threshold=hc['pthresh'])
            # train_output ALREADY contains calculated metrics
            
            # Dev pass - DON'T call calculate_metrics again
            dev_output = train_dev_pass(lsc_model, optimizer, dev_loader, sec_batch, metrics=dev_mlmetrics, pred_threshold=hc['pthresh'])
            # dev_output ALREADY contains calculated metrics
            
            # Use the returned metrics objects
            train_loss, dev_loss = train_output.loss, dev_output.loss

            if dev_loss < best_loss:
                best_loss = dev_loss
                best_metrics = dev_output
                best_model = lsc_model.state_dict()
                
            scheduler.step(dev_loss)
                
            print("%5d || %.4f | %.4f || %.4f | %.4f %.4f %.4f" % (
                epoch, 
                train_output.loss, train_output.macro_f1, 
                dev_output.loss, dev_output.macro_prec, dev_output.macro_rec, dev_output.macro_f1
            ))
        print("\nCollecting outputs")
        print("==================================================")
        
        # FIX 2: Create parent directory before saving the model (solves the RuntimeError)
        if not os.path.exists(os.path.dirname(dc['model_dump'])):
            os.makedirs(os.path.dirname(dc['model_dump']))
            
        torch.save(best_model, dc['model_dump'])
        with open(dc['dev_metrics_dump'], 'wb') as fw:
        	pkl.dump(best_metrics, fw)

        if hc['do_test']:
            lsc_model.load_state_dict(best_model)
            
        if use_kdtree:
            sec_struct_embs = lsc_model.graph_encoder.node_embedding['section'].weight.data.cpu().numpy()
            kdtree.build(sec_struct_embs)
            
        # Updated Printout for Validation Results
        print("\nVALIDATION Results")
        print("Loss: %.4f | Macro P: %.4f Macro R: %.4f Macro F1: %.4f" % (best_loss, best_metrics.macro_prec, best_metrics.macro_rec, best_metrics.macro_f1))

    if hc['do_test']:
        test_mlmetrics = MultiLabelMetrics(L, dev=str(DEVICE))
        print("\nRunning Test")
        print("==================================================")
        test_output = train_dev_pass(lsc_model, optimizer, test_loader, sec_batch, metrics=test_mlmetrics, pred_threshold=hc['pthresh'])
        test_mlmetrics.calculate_metrics()
        
        with open(dc['test_metrics_dump'], 'wb') as fw:
            pkl.dump(test_mlmetrics, fw)
        # Updated Printout for Test Results
        print("TEST Results")
        print("Loss: %.4f | Macro P: %.4f Macro R: %.4f Macro F1: %.4f" % (test_mlmetrics.loss, test_mlmetrics.macro_prec, test_mlmetrics.macro_rec, test_mlmetrics.macro_f1))

    if hc['do_infer']:
        print("\nRunning Inference")
        print("==================================================")
        
        if use_kdtree and kdtree and kdtree.built:
            print(f"Using KD-Tree with k={k_neighbors} for fast retrieval.")
        else:
            print("KD-Tree is inactive or not built. Running full scoring.")
            kdtree = None
            
        infer_outputs = train_dev_pass(
            lsc_model, 
            optimizer, 
            infer_loader, 
            sec_batch, 
            infer=True, 
            pred_threshold=hc['pthresh'], 
            label_vocab=label_vocab,
            kdtree=kdtree
        )
        
        with open(dc['infer_trg'], 'w') as fw:
            # Note: json.dumps requires serializable objects, ensure 'obj' is used instead of 'list'
            fw.write('\n'.join([json.dumps(obj) for obj in infer_outputs]))


if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main_execution()
