# FAFNIR: Accelerating Sparse Gathering by Using Efficient Near-Memory Intelligent Reduction

## Challenges of Embedding Lookup (Sparse Gathering)
- Data movement
  - (# queries)x(# vectors in a query)x(vector size) elements must be tranferred from memory systems to cores: unscalable
  - Goal: reduce the amount of data movement
- Lack of utilization of row buffer locality
  - TensorDIMM
    - column-major order, which breaks the row-buffer locality in the DRAM system
    - splitting embedding vectors across more ranks causes poor utilization of row-buffers
      - we must open a row, but read a "smaller fraction" of it
    - more ranks -/-> increase rank-level parallelism
  - RecNMP
    - good: more ranks --> increase rank-level parallelism, rank-level parallelism O
    - bad: if low spatial locality, parallel compute at NDP is low compared to that of in theory, DIMM-level parallelsim X
- Relying on spatial locality
  - RecNMP
    - problem: the probability of having a query with indices on the same channel(a group of DIMMs) is only up to 25% in a four-channel system
    - if the vectors in each query reside in different DIMM, scalar operation in NDP goes to 0 which means all operations should be done in cores after transfer stage
    - = raw data needs to be transferred to the cores, so memory bandwidth may not be fully utilized
- Connection overhead
  - previous studies: # of connections = (# of cores)x(# of memory devices) -> costly and limit the scalability
- Using caches to reduce memory accesses
  - RecNMP: use cache at NDP -> 128KB HW overhead
<p align="center"> <img width="1327" alt="image" src="https://user-images.githubusercontent.com/49300363/127852718-5a944b75-2d3c-4d7c-9e57-dcd98d11308c.png"> </p>


## Contributions
- FAFNIR: process data while it is gathered
  - Use "reduction tree", the leaves of which are connected to the ranks of a memory system and the nodes are PEs
  - Low data movement and few connections: cxm -> 2m-2(intermediate connections) + c(to cores) -> scalable
  - High-level parallelism and scalability: rank-level parallelism -> DIMM-level parallelism
  - Effective memory-access reduction: read only unique memory access & no cache mechanism
  - Applicable to other sparse problem: SpMV (Sparse matrix-vector multiplication) for genericity
<p align="center"><img width="800" alt="image" src="https://user-images.githubusercontent.com/49300363/127853725-231da25d-d62e-4d66-a384-cbf637260c9f.png"> </p>
  
  - 4 DIMMs/channel, 2 ranks/DIMM -> 4 channels, 4 DIMM/rank nodes, 1 channel nodes

## FAFNIR
- Microarchitecture of PE
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127854382-ee0b9ced-b067-47e7-a34a-47d74bfdbe95.png"></p>

- Key machanisms
<p align="center"><img width="1475" alt="image" src="https://user-images.githubusercontent.com/49300363/127854578-cdbb0520-f258-4601-a92b-4604a279f234.png"></p>
  
  - (b): "remaining indices" of queries contating "Inx" (uniques set of indices)
    - e.g., Query a and c have index 11 -> queries of headers = a(32, 83, 77) + c(50, 94, 26)
  - (c): reduce = if there is a query among B[x].queries which is same as A[i].indices, do reducing
    - out.indices = concat(A[i].indices, B[x].indices)
    - out.queries = exclude(B[x].queries, A[i].indices) (= B[x].quries - A[i].indices)
    - e.g., out.indices = concat(11, 50), out.queries = exclude([11, 94, 26], 11)

- Adapting Fafnir to SpMV
  - Main difference between Farnir & SpMV
    - we do not know where the non-zero values of the sparse matrix are located = indices are unknown
    - we need to reduce the elements of a vector into one element
    - vectorization is required
    - make sparse matrix as linked list (list of list, LIL)
    - SpMV: only leaf PEs multiply data vs. Embedding lookup: skip multiplication
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127856526-af41dbeb-caf2-4fe8-b745-e7d180fa9928.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127856668-1e70bb53-d3fa-416f-a1c4-97ee3639faaa.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127857321-c53373a1-03f9-4c37-8394-ff467a695608.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/49300363/127857360-75e3e3b6-3961-4f82-b124-07115b26a808.png"></p>

## Evaluation
- C++ -> RTL generation, Performance evaluation
- Vivado HLS: for synthesis & implementation, on XCVU9P FPGA targeting a VCU1525 acceleration development kit (16GB DDR4 DIMMs = 64GB total per DIMM/rank node)
- Application: recommendation system (embedding lookup), graph analytics and scientific applications (SpMV)
- TBU
