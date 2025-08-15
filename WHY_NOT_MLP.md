# Why PointFlow Decoder > MLP Decoder

## MLP Decoder Problems for Point Clouds:

### 1. Fixed Output Size
```python
# MLP forces you to pick N points upfront
mlp = nn.Sequential(
    nn.Linear(latent_dim, 512),
    nn.ReLU(),
    nn.Linear(512, N * 2)  # N is FIXED!
)
```
**Problem**: Real slices have variable point counts (100-2000+)

### 2. No Permutation Invariance
```python
# MLP output depends on point order
points = mlp(z).reshape(N, 2)
# Shuffle input → completely different output!
```
**Problem**: Point clouds have no inherent order

### 3. Poor Spatial Modeling
- MLP treats output as one big vector
- No understanding of point relationships
- Can't model local structures well

## PointFlow Decoder Advantages:

### 1. Variable Point Counts
```python
# Generate ANY number of points
N = predict_point_count(z)  # Adaptive!
y = torch.randn(N, 2)        # Sample N points
x = flow.decode(y, z)        # Transform to slice
```

### 2. Continuous Distribution
- Models probability density over R²
- Can sample as many/few points as needed
- Natural handling of shape complexity

### 3. Invertible by Design
```python
# Forward: slice → Gaussian (for training)
y = flow.encode(x, z)

# Reverse: Gaussian → slice (for generation)  
x = flow.decode(y, z)
```

### 4. Spatial Understanding
- CNF learns smooth transformations
- Preserves local geometry
- Models complex boundaries naturally

## Visual Comparison:

```
MLP Decoder:
z → [||||||||] → 1024 numbers → reshape → points
    (black box)   (no structure)

PointFlow Decoder:
z → CNF → smooth transformation → point cloud
           (preserves geometry)
```

## Why This Matters for Your Project:

1. **Real automotive slices**: 100-2000+ points (highly variable)
2. **Complex shapes**: Need to model curves, not just point positions
3. **Generation quality**: CNF produces smoother, more realistic shapes
4. **Future flexibility**: Can adapt point count per slice

## Bottom Line:
MLP is a toy solution. PointFlow is the robust solution you need.
