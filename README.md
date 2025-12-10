# Chess Training Data - 280K Positions

Fine-tune Qwen2.5-1.5B-Instruct on ~280K chess positions.

## Files

- `train_qwen_1.5b_a100.ipynb` - Training notebook for A100 GPU
- `merge_data.py` - Script to merge puzzle and SF position data
- `stockfish_labeler.py` - Stockfish labeling script

## Data

The training data file (`train_data_280k.jsonl`, 172MB) is too large for GitHub.

**Download from:** [Add your link here after uploading]

Or generate it yourself using the data pipeline in the main repo.

## Data Format

```json
{
  "messages": [
    {"role": "user", "content": "You are an expert chess player..."},
    {"role": "assistant", "content": "<think>White pawn from e2 moves to e4.</think><uci_move>e2e4</uci_move>"}
  ]
}
```

## Training

1. Upload `train_data_280k.jsonl` to your GPU instance
2. Open `train_qwen_1.5b_a100.ipynb`
3. Configure settings (LoRA vs full fine-tuning)
4. Run all cells

### Recommended Settings

| Setting | LoRA | Full FT |
|---------|------|---------|
| VRAM | ~16GB | ~30GB |
| Time (3 epochs) | ~2-3 hrs | ~4-6 hrs |
| Quality | Good | Better |

## Dataset Composition

- **100K** Lichess puzzles (hanging piece focus)
- **179K** Stockfish vs Stockfish positions (diverse openings/middlegames)
