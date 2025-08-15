# Project Roadmap: Text → 3D via Cross-Sections

## Phase 1: Autoencoder (Current) ✅
**Goal**: Encode/decode 2D slices reliably
**Timeline**: 1 week
- [x] Implement PointNet2D encoder
- [x] Implement PointFlow decoder
- [ ] Two-stage training on single slice
- [ ] Scale to 10 cars dataset
- [ ] Full dataset training

## Phase 2: Transformer Training 🚀
**Goal**: Predict next slice embedding
**Timeline**: 2-3 weeks
- [ ] Implement positional encoding for slice order
- [ ] Transformer architecture (GPT-style)
- [ ] Training on slice sequences
- [ ] Validation on held-out shapes

## Phase 3: Conditional Generation 🎨
**Goal**: Text/image → 3D shape
**Timeline**: 2-3 weeks
- [ ] Add CLIP text encoder
- [ ] Condition transformer on prompts
- [ ] Fine-tune on text-shape pairs
- [ ] Deploy demo

## Why This Works:
1. **Proven components**: PointNet, PointFlow, Transformers all work
2. **Clear data flow**: Text → Embeddings → Slices → 3D
3. **Incremental validation**: Test each phase before moving on
4. **Not reinventing the wheel**: Using established architectures

## Simplified Timeline:
- **Week 1-2**: Get autoencoder working (YOU ARE HERE)
- **Week 3-4**: Scale to full data
- **Week 5-7**: Add transformer
- **Week 8-10**: Add conditioning & polish

Total: ~2.5 months to working prototype!
