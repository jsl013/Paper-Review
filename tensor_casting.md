# Tensor Casting: Co-Designing Algorithm-Architecture for Personalized Recommendation Training
## Limitation of state-of-the-art
- TPU for DNN training: General-purpose matrix multiplication (GEMM) because of the versatility of GEMM operators
  - However, NLP, recommendation system's "sparse" embedding layers significantly differ from "dense" DNN layers
- Embedding table: memory intensive (capacity & bandwidth) -> use local memory instead of HBM near GPU (size: tens of GB) -> CPU-centric training of embeddings
  - Embedding layer -> CPU, MLP -> GPU
  - CPU is bottleneck :(
## Target
- Backpropagation stage of embedding layer in recommendation system
  - Key Idea: gradient expand-and-coalesce -> tensor gather-and-reduce (forward prop.)
## Contributions
- Tensor Casting: cast the gradient expand-and-coalesce operation to the tensor gather-and-reduce operator
## Background
- Recommendation system
<img width="1211" alt="image" src="https://user-images.githubusercontent.com/49300363/127619087-e5e7d7ae-264e-4a2a-896c-fcd7de0f670e.png">
<p align="center"> <img width="565" alt="image" src="https://user-images.githubusercontent.com/49300363/127619384-0a7aecdc-423a-40c6-bca9-7bd39b1bc61f.png">

- High-level overview of tensor gather-reduce & gradient expand-coalesce-scatter operation
  - (a): forward propagation, (b): backpropagation of embedding layers (batch size=2)
  <img width="886" alt="image" src="https://user-images.githubusercontent.com/49300363/127619603-9c062947-32c7-4aac-a310-a0f6e9aec517.png">
  
  - Gradient coalescing
  <img width="634" alt="image" src="https://user-images.githubusercontent.com/49300363/127620298-1360910b-dca3-4115-a462-68fc27233e0f.png">
