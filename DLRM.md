# Factorization Machine

## Limitation of state-of-the-art
1) SVM: not good at sparse data (recommendation system)
2) Other factorization models: specialized for certain prediction task

## Advantage of factorization machine
1) good parameter estimation under sparse data
2) computation: linear time complexity
3) generalized predictor

## Specific Description of Algorithm
### Intuitive Description
<p align="center"><img width="468" alt="image" src="https://user-images.githubusercontent.com/49300363/127447291-823671a6-a70e-460c-9544-26dc4a4ce59c.png">

- User: A = Alice, B = Bob, C = Charlie
- Movie: TI = Titanic, NH = Notting Hill, SW = Star Wars, ST = Star Trek
- Target y = rate of the movie from the user
    - e.g. x^((1)) = Alice rated Titanic as point 5

Intuition #1. If two users have similar preference to certain item(s), the users would have similar preference
     => e.g. Bob and Charlie will have similar factor vectors v_B,v_C because both have similar interactions with Star Wars (v_SW) for predicting ratings – i.e. <v_B,v_SW> and <v_C,v_SW> have to be similar

Intuition #2. If two users have different preference to certain item(s), the users would have different preference
    => e.g. Alice (v_A) will have a different factor vector from Charlie (v_C) because she has different interactions with the factors of Titanic and Star Wars for predicting ratings
Intuition #3.  If many users have similar preference tendency to certain items, those items would have similar characteristics
    => e.g. The factor vectors of Star Trek are likely to be similar to the one of Star Wars because Bob has similar interactions for both movies for predicting y. In total, this means that the dot product (= interaction) of the factor vectors of Alice and Star Trek will be similar to the one of Alice and Star Wars – which also makes intuitively sense.

### Computation Description
- Computation: linear time complexity O(kn)=O(km ̅_D) (m ̅_D: average # of non-zero elements)
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/49300363/127447337-7c32912b-75e0-40ae-b0cd-9c1f21502939.png">
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/49300363/127447350-605ba8f9-b11b-421a-898c-6709969e8232.png">

- Learning method: SGD (stochastic gradient descent) -> learning w,V
<p align="center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/49300363/127447434-8eb7553b-944d-4b82-841a-8de701e3be51.png">

## Comparison
- SVM: all interaction parameters w_(i,j) of SVMs are completely independent (= no interaction)
    => e.g. w_(i,j),w_(i,l) are independent but in FM, <v_i,v_j> and <v_i,v_l> are dependent by shared parameter v_i
- Other models: FM is similar to other specific-purposed models, so FM is a generalized model
## Questions
Q1. What is the main difference between factorization machine and matrix factorization?
A1. FM = MF (interaction btw. user & item) + biases for user & item

# DLRM
- Model Overview of DLRM
 
- Reference:
https://www.kdnuggets.com/2021/04/deep-learning-recommendation-models-dlrm-deep-dive.html

## Main bottleneck of DLRM = Embedding table size & lookup
- Embedding
  - Direct (naïve) Embedding: Memory bandwidth & capacity constraint (embedding table lookup is irregular memory access pattern)
  - Hashing: Use hash function, but there is hash collision! (It is not a unique mapping)
  - Compositional Embedding: Reduce size of embedding table, make complementary partitions and mapping each partition to one embedding table -> combine all embedding vectors into one
      - Operation-Based
        - concatenation
        - addition
        - element-wise multiplication
      - Path-Based
        - linear function
        - MLP: if # of added training parameters are more than embedding table size, …
      - Feature Generation: directly use as separate sparse features (embedding vectors from tables), no other operation is needed

- Q. How to make complementary partitions?
    - Naïve partition
    - Quotient - remainder partitions
    - Generalized quotient - remainder partitions
    - RNS partitions (Chinese remainder partitions)
 
## Parallelism
- Data parallelism for MLP + model parallelism for embedding table
    - model parallelism has been used to distribute the embeddings across devices
    - PyTorch & Caffe2 didn’t provide native support for model parallelism 
        - PyTorch: nn.EmbeddingBag
        - Caffe2: SparseLengthSum
- Butterfly shuffle for the all-to-all (personalized) communication
    - appropriately slice the resulting embedding vectors and transfer them to the target devices
 
## Reference
- Latent factor & Matrix factorization: https://wooono.tistory.com/149

