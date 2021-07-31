# Tensor Casting: Co-Designing Algorithm-Architecture for Personalized Recommendation Training
## Limitation of state-of-the-art
- TPU for DNN training: General-purpose matrix multiplication (GEMM) because of the versatility of GEMM operators
  - However, NLP, recommendation system's "sparse" embedding layers significantly differ from "dense" DNN layers
- Embedding table: memory intensive (capacity & bandwidth) -> use local memory instead of HBM near GPU (size: tens of GB) -> CPU-centric training of embeddings
  - Embedding layer -> CPU, MLP -> GPU
  - CPU is bottleneck :(
## Target
- Backpropagation stage of embedding layer in recommendation system
  - Key Idea: gradient expand-and-coalesce = bottleneck of training recommendations, so cast gradient expand-and-coalesce -> tensor gather-and-reduce(forward prop.) (=Tensor Casting), and accelerate Tensor gather-reduce operator!
## Contributions
- Tensor Casting: cast the gradient expand-and-coalesce operation to the tensor gather-and-reduce operator
- NMP microarchitecture for accelerating tensor gather-reduce operator
## Background
- Recommendation system
<p align="center"> <img width="614" alt="image" src="https://user-images.githubusercontent.com/49300363/127619087-e5e7d7ae-264e-4a2a-896c-fcd7de0f670e.png"> </p>
<p align="center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/49300363/127619384-0a7aecdc-423a-40c6-bca9-7bd39b1bc61f.png"> </p>

- High-level overview of tensor gather-reduce & gradient expand-coalesce-scatter operation
  - (a): forward propagation, (b): backpropagation of embedding layers (batch size=2), (c)(third img): Tensor Casting + Gather-reduce + Scatter, (d)(forth img): Tensor Casting
  - Key idea of tensor casting: separate sorting & accumulate, so latency of sorting can be hidden in forward stage because sorting part can be done before finishing forward propagation
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127619603-9c062947-32c7-4aac-a310-a0f6e9aec517.png">
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127726524-981108fe-a231-4c2e-bbd7-c43189c087bc.png">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127726546-8d894ef6-4f0e-47ec-8d9b-5cb5f13bfd6d.png">

  - Gradient coalescing & Tensor Casting
    - Gradient coalescing: Sorting + Accumulate
    - Tensor Casting: Sorting
  
  <img align="left"><img width="634" alt="image" src="https://user-images.githubusercontent.com/49300363/127620298-1360910b-dca3-4115-a462-68fc27233e0f.png">
  <img align="right"><img width="634" alt="Screen Shot 2021-07-31 at 11 47 27 AM" src="https://user-images.githubusercontent.com/49300363/127726429-72b33d56-81d6-44d5-82a4-4bf4fc11c28f.png">
  <img width="634" alt="image" src="https://user-images.githubusercontent.com/49300363/127726591-8f839395-f28d-4e40-9a82-2f0423292cb8.png">
## Performance Improvement
- CPU-centric < CPU-centric + Tensor Casting < Memory-centric(NMP) + Tensor Casting
  <img width="614" alt="image" src="https://user-images.githubusercontent.com/49300363/127726664-2a036dbf-01eb-4c01-8ae4-2d3ff6998162.png">
- Proposed memory-centric sytem for trainig recommendation system
  - NMP = accelerating tensor gather-reduce operation
    - The design can cover the majority of embedding training time using a single
  <img width="614" alt="image" src="https://user-images.githubusercontent.com/49300363/127726748-47a8f911-17d3-4eeb-99ec-7bdc0dbd8de2.png">
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/49300363/127726751-d4350ec6-6ef3-4d59-8688-0cc31a23552a.png">
  
  - NMP microarchitecture
    - vector ALU: reduce the gathered embeddings
    - a local memory controller: translate the tensor gather-reduce and scatter instructions into low-level DRAM commands
    - a set of input/output buffers: stage in/out the embedding vectors or other meta data for gather-scatter
    - assumption: GPU ---CISC instruction---> NMP
    - NMP core = per rank (granularity = 64B), rank-level parallelism for bandwidth amplification
    - advantage: no data movement outside the DIMMs -> significant increase in the aggregate effective memory bandwidth (?)
    <img width="627" alt="image" src="https://user-images.githubusercontent.com/49300363/127727359-95599d9f-1e74-4809-9c98-ff0589fc3edc.png">
    <img width="627" alt="image" src="https://user-images.githubusercontent.com/49300363/127727416-591d362f-9f22-4180-9884-cf5b403580be.png">

    
## Evaluation
- Low design overhead
  - the modifications required in the DIMM or the underlying NMP microarchitecture is practically negligible as the key innovation of Tensor Casting is our permutation algorithm itself that enable all major compute primitives of training embeddings to operate over the tensor gather-scatter accelerator. The primary change required is the inclusion of the tensor scatter instruction as part of the ISA.
- Latency breakdown of Tensor casting compared to baseline(CPU, NMP)
<img width="1565" alt="image" src="https://user-images.githubusercontent.com/49300363/127727475-636938d9-2499-47ab-b6c2-24540b48ed43.png">
- Performance speedup & Energy consumption
<img width="1613" alt="image" src="https://user-images.githubusercontent.com/49300363/127727488-e519a2be-dbc0-4bea-93a2-396e4c3a4e92.png">
- NMP uilization
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127727511-1448688a-f588-43b8-8106-8d408c7fb2c9.png">
- Robustness of Tensor Casting according to batch size
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127727539-a5783158-cc03-43df-8a56-a688701839c8.png">
- Robustness of Tensor Casting according to embedding vector size
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127727550-f934e227-02cb-4c2d-899a-a4e320379c0f.png">







