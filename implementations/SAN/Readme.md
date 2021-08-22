## Pytorch Implementation of [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274.pdf)


project moved to [kaggle](https://www.kaggle.com/harshraj22/stacked-attention-network/)    
### 1. Architectural overview:
> ##### 1.1 Image Features:
> Image features are extracted using a CNN, and the features (v<sub>i</sub>) are arranged to form a `m x h` matrix.
> ![A3282E37-3636-4209-9C86-DA8BF62D2191_4_5005_c](https://user-images.githubusercontent.com/46635452/129331784-e221bd8f-7923-4298-a9c7-1606e857789a.jpeg)

> #### 1.2 Question Features:
> Question features are extracted using LSTM, giving a final output vector (v<sub>q</sub>) of shape `h`.

> <img src="https://user-images.githubusercontent.com/46635452/129332949-65574cab-1fd3-461c-96fe-bca4f1d05ff6.png" height="300">

> #### 1.3 Attention:
> ![image](https://user-images.githubusercontent.com/46635452/129433854-0c223aa5-33d8-42db-b8cd-c8ee9bd3e128.png)

> Attention is calculated, with query as the question feature vector (v<sub>q</sub>) and the image features (v<sub>i</sub>) as values. The weighted sum of values gives a new vector of shape `h`. This is used as question feature vector (v<sub>q</sub>) for calculating attention in next step.
> Successive attention scores are calculated, and the question feature vector (v<sub>q</sub>) keeps on getting updated, adding more information from image feature vectors to itself.

> #### 1.4 Word Generation:
> The final question vector (v<sub>q</sub>), which contains information from image feature vector, is passed through a Linear layer, giving output vector of shape `vocab_len`. The predicted word is found by taking the argmax of the output feature vector.


### 2. Training & Results:
<img src="https://user-images.githubusercontent.com/46635452/130365587-6138c01a-6725-4416-8d2b-f1175b07f34f.png" > </img>
