#!/usr/bin/env python3
"""Exercise original retrieval functions on a few real, complete candidate graphs.

All generated caches/checkpoints live in a new output directory. Original model
weights are loaded, updated for one small epoch, and never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'retrieval'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=['webqsp', 'cwq'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--identifiers', type=Path, required=True)
    parser.add_argument('--hf-cache-root', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from src.dataset.emb import EmbInferDataset
    from src.dataset.retriever import RetrieverDataset, collate_retriever
    from scripts.expand_webqsp_scored_triples import (
        resolve_hf_dataset_dir, load_hf_splits, build_reasoning_paths_for_sample,
        build_output_item,
    )

    identifiers = set(args.identifiers.read_text().splitlines())
    raw_path = output / 'raw_samples.json'
    if raw_path.exists():
        raw_splits = json.loads(raw_path.read_text())
    else:
        if args.dataset == 'cwq':
            cache = resolve_hf_dataset_dir(args.dataset, args.hf_cache_root)
            source = load_hf_splits(args.dataset, cache)
        raw_splits = {}
        for split, count in [('train', 2), ('validation', 1), ('test', 1)]:
            if args.dataset == 'webqsp':
                url = ('https://datasets-server.huggingface.co/rows?dataset=ml1996%2Fwebqsp'
                       f'&config=default&split={split}&offset=0&length=20')
                response = subprocess.run(['curl', '-fLsS', '--max-time', '60', url],
                                          check=True, capture_output=True, text=True)
                payload = json.loads(response.stdout)
                candidates = [item['row'] for item in payload['rows']
                              if not item.get('truncated_cells')]
            else:
                candidates = [source[split][i] for i in range(min(100, len(source[split])))]
            raw_splits[split] = []
            processor = EmbInferDataset.__new__(EmbInferDataset)
            scorer = RetrieverDataset.__new__(RetrieverDataset)
            # Prefer small but complete real graphs; do not truncate their edges.
            for row in sorted(candidates, key=lambda x: len(x['graph'])):
                sample = processor._process_sample(row, identifiers)
                if not sample['q_entity_id_list'] or not sample['a_entity_id_list']:
                    continue
                scores, length = scorer._extract_paths_and_score(sample)
                if length is None or length == 0 or not scores.any():
                    continue
                raw_splits[split].append(row)
                if len(raw_splits[split]) == count:
                    break
            if len(raw_splits[split]) != count:
                raise RuntimeError(f'Not enough usable real graphs for {split}')
        raw_path.write_text(json.dumps(raw_splits, ensure_ascii=False, indent=2))
    print({s: [(r['id'], len(r['graph'])) for r in rows] for s, rows in raw_splits.items()}, flush=True)
    if args.prepare_only:
        return

    from src.model.text_encoders import GTELargeEN
    from src.model.retriever import Retriever
    from src.setup import set_seed
    from train import train_epoch, eval_epoch
    from inference import infer_split, to_jsonable
    from torch.utils.data import DataLoader

    assert torch.cuda.is_available(), 'This smoke test expects the selected GPU'
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    cpt = torch.load(args.checkpoint, map_location='cpu')
    config = cpt['config']
    assert config['dataset']['name'] == args.dataset
    assert config['dataset']['text_encoder_name'] == 'gte-large-en-v1.5'
    set_seed(config['env']['seed'])
    os.chdir(output)
    data_root = output / 'data_files' / args.dataset
    (data_root / 'processed').mkdir(parents=True, exist_ok=True)
    emb_root = data_root / 'emb' / 'gte-large-en-v1.5'
    emb_root.mkdir(parents=True, exist_ok=True)
    encoder = GTELargeEN(device)
    for split, rows in raw_splits.items():
        name = 'val' if split == 'validation' else split
        subset = EmbInferDataset(rows, identifiers, str(data_root / 'processed' / f'{name}.pkl'))
        cache = {}
        for sample_id, question, entities, relations in subset:
            def batched(texts):
                return torch.cat([encoder.embed(texts[i:i+32]) for i in range(0, len(texts), 32)]) if texts else encoder.embed([])
            cache[sample_id] = dict(q_emb=encoder.embed([question]),
                                    entity_embs=batched(entities), relation_embs=batched(relations))
        torch.save(cache, emb_root / f'{name}.pth')
    del encoder
    torch.cuda.empty_cache()
    subsets = {s: RetrieverDataset(config, s, skip_no_path=False) for s in ['train', 'val', 'test']}
    model = Retriever(1024, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'], strict=True)
    before = model.pred[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    train_stats = train_epoch(device, DataLoader(subsets['train'], batch_size=1, collate_fn=collate_retriever), model, optimizer)
    delta = (model.pred[0].weight.detach() - before).abs().max().item()
    assert torch.isfinite(torch.tensor(train_stats['loss'])) and delta > 0
    val_stats = eval_epoch(config, device, DataLoader(subsets['val'], batch_size=1, collate_fn=collate_retriever), model)
    torch.save({'config': config, 'model_state_dict': model.state_dict()}, output / 'smoke_cpt.pth')
    # Strict reload verifies the saved retrieval model before inference.
    model.load_state_dict(torch.load(output / 'smoke_cpt.pth', map_location=device)['model_state_dict'], strict=True)
    model.eval()
    merged = output / 'merged'
    merged.mkdir(exist_ok=True)
    path_counts = {}
    for split, subset in subsets.items():
        predictions = infer_split(split, subset, model, device, max_K=500)
        (output / f'{split}.json').write_text(json.dumps(to_jsonable(predictions), indent=2))
        raw_split = 'validation' if split == 'val' else split
        rows = {r['id']: r for r in raw_splits[raw_split]}
        items = []
        for sample_id, prediction in predictions.items():
            prediction['id'] = sample_id
            paths = build_reasoning_paths_for_sample(prediction, rows[sample_id], 1/3, 3)
            assert paths, f'No recovered paths for {sample_id}'
            items.append(build_output_item(prediction, rows[sample_id], paths))
        (merged / f'{raw_split}.json').write_text(json.dumps(items, ensure_ascii=False, indent=2))
        path_counts[raw_split] = [len(x['reasoning_paths']) for x in items]
    report = dict(dataset=args.dataset, source_checkpoint=str(args.checkpoint),
                  text_encoder='gte-large-en-v1.5', embedding_dim=1024,
                  optimizer_steps=len(subsets['train']), train=train_stats,
                  max_weight_change=delta, validation=val_stats,
                  recovered_path_counts=path_counts, status='passed')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
