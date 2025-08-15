# Decoder Debugging Summary

## What We Learned

### Test 1 Results (test_decoder_only.py)
- **Loss plateau**: 0.63 → 0.24 (stuck there)
- **Problem**: Learning rate too low, no scheduling
- **Conclusion**: Decoder CAN learn but needs better hyperparameters

## The Issues

1. **Learning Rate**: 1e-4 is too conservative
2. **No LR Scheduling**: Gets stuck in local minima
3. **Architecture Complexity**: hidden_dim=256 might be overkill
4. **ODE Tolerances**: Still might be too strict

## Three Test Scripts (In Order)

### 1. test_decoder_simple.py (Start Here!)
- Uses only 50 points (easier problem)
- Smaller architecture (hidden_dim=64)
- Higher learning rate (1e-2)
- Quick test (~2 mins)

```bash
python scripts/test_decoder_simple.py data/ --slice-name single_slice_test.npy
```

### 2. test_decoder_v2.py (If simple works)
- Full 584 points
- Better hyperparameters:
  - LR: 5e-3 with cosine scheduling
  - Hidden dim: 128
  - ODE tolerances: 5e-3
- Better visualizations

```bash
python scripts/test_decoder_v2.py data/ --slice-name single_slice_test.npy
```

### 3. Two-Stage Training (If v2 works)
- Use the working hyperparameters
- Train full VAE properly

## Decision Tree

```
Run test_decoder_simple.py
    ↓
Loss < 0.1? → YES → Run test_decoder_v2.py
             ↓                ↓
             NO               Loss < 0.15? → YES → Proceed to two-stage
                                          ↓
                                          NO → Debug architecture

If simple test fails:
- The CNF architecture has fundamental issues
- Consider alternative decoder architectures
```

## Key Insights

1. **CNF can learn** - just needs right hyperparameters
2. **Start simple** - prove it works on easy problems first
3. **Learning rate matters** - 1e-4 was way too low
4. **Scheduling helps** - prevents getting stuck

## Recommended Next Steps

1. Run `test_decoder_simple.py` (2 mins)
2. If it works, run `test_decoder_v2.py` (5 mins)
3. If both work, implement two-stage training with these hyperparameters
4. If simple fails, we need to reconsider the architecture
