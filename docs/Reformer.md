# Reformer
## Limitation of state-of-the-art
- Memory problem of attention layer
  - Scaled dot-product attention
  - Time complexity: O(L^2), L = # of input data tokens
  - If input data is large book or something long, it could be a problem!
<p align="center"> <img width="280" alt="image" src="https://user-images.githubusercontent.com/49300363/127449360-6f1c2f94-bd06-456a-bb4d-5f3eea34f961.png">

  - Feed-forward layer
    - The size of weight matrix of FF layer is large enough
    - In common case, the size of input & output dimension of feed-forward is much larger than size of embedding vector
  - Stacked residual connection
    - For backpropagation, non-reversible network need to store the activations of layers :(
    - N-layer of attention + FF = Huge memory consumption...
  
## Contributions
<img width="1483" alt="image" src="https://user-images.githubusercontent.com/49300363/127456692-1e117fcb-8406-42db-ae00-2f8c2eb13a6f.png">

- Approximate attention = Locality-sensitive hashing (LSH)
  - Instead of compute total matrix-matrix multiplication, compute softmax only for similar pairs of words
    - large number gets larger after softmax, but small number gets much smaller (=no need to caculate)
  - We can find similar pairs(buckets) of words using LSH
    - There is a small probability that similar items nevertheless fall in different buckets, so they used "multi-round LSH attention"
  - Key and Query is same (Q = K)
  - Steps
    - Color = Hash bucket
    <img width="475" alt="image" src="https://user-images.githubusercontent.com/49300363/127455228-79924a3e-9d60-465c-acc1-663e5216027b.png">
    <img width="475" alt="image" src="https://user-images.githubusercontent.com/49300363/127455276-61704504-a127-43b4-9694-dc76898ecc9f.png">
    <img width="475" alt="image" src="https://user-images.githubusercontent.com/49300363/127455400-a9b6ff7a-dcc7-4beb-b3e0-520dbf36ce4a.png">
![image](https://user-images.githubusercontent.com/49300363/127456822-bb09b466-daf4-4c3a-9ee5-e97550a45cfe.png)

- Reversible layer
![image](https://user-images.githubusercontent.com/49300363/127452352-73d074cf-8367-4928-99a2-f4c21250bacb.png)
  - Equation of original residual layer and reversible residual layer
    - Using first equation, we cannot recover x from y, so we have to store the x
    - However, we can recover x from y using second and third equations
<p align="center"><img width="110" alt="image" src="https://user-images.githubusercontent.com/49300363/127452989-ac4713d8-e89a-4a8f-a4ac-443011ef9ae6.png">
<p align="center"><img width="300" alt="image" src="https://user-images.githubusercontent.com/49300363/127453344-dd51ec5e-ddd9-400f-a8c9-63dbc8d96aca.png">
<p align="center"><img width="300" alt="image" src="https://user-images.githubusercontent.com/49300363/127453757-13136992-eb2d-4ccc-be6b-c1adcbb25080.png">
<p align="center"><img width="447" alt="image" src="https://user-images.githubusercontent.com/49300363/127454855-d65e9ada-8fff-437d-a47d-526de58f9ce7.png">

- Chunking
  - While using reversibility, the thicker layers can still use a lot of memory
  - Feed-forward layer in particular can use intermediate vectors of dimensionality d_ff=4K or highr
  - However, computatoins in FF layers are completely independent across positoins in a sequence, so the compuatation can be split into c chunks
  <img width="600" alt="image" src="https://user-images.githubusercontent.com/49300363/127456375-7094f51e-4ce6-4483-9e6d-6f60f3edb0ae.png">


## Reference
- Reformer: https://arxiv.org/pdf/2001.04451.pdf
- Reformer review: https://blog.pingpong.us/reformer-review/
