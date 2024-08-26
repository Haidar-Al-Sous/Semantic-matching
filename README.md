# Comparative Study of Semantic Matching Techniques

## Introduction

We evaluated different embeddings techniques using this [dataset](https://www.kaggle.com/competitions/quora-question-pairs) in the context of semantic matching. We used six techniques:
1. **Unweighted Average of Word2Vec Embeddings.**
2. **Unweighted Average of Siamese CBOW Embeddings.**
3. **Sentence-BERT Embeddings.**
4. **SIF Weighted Average of GloVe Embeddings.**
5. **Sent2Vec Embeddings.**
6. **Doc2Vec Embeddings.**

Before explaining each appproach, we want to emphasize that the goal is to calculate sentence embeddings in the most accurate way. After that, we train a logistic regression model to output the similarity based on the calculated embeddings from each approach.

### 1. Unweighted Average of Word2Vec Embeddings
This approach consists of simply taking average of words' embeddings (using Word2Vec) for each sentence.

### 2. Unweighted Average of Siamese CBOW Embeddings [1]
In this approach, the goal is to compute word embeddings in a way that uses information about nearby sentences. For each sample, we consider five sentences: the current one, the previous one, the next one, and two randomly selected ones. We used the Brown corpus as our training data.

![Network Architecture](https://github.com/user-attachments/assets/5d78ebac-e763-424e-b229-d55bf88dc717)

**In the 1st layer,** we calculate the average of word embeddings for each sentence.  
**In the 2nd layer,** we compute the cosine similarity score between each pair of sentences.  
**In the 3rd layer,** we apply the softmax function to each score from the previous layer.  
We update the word embeddings based on stochastic gradient descent.

### 3. Sentence-BERT Embeddings
We used `FastSentenceTransformer` to compute sentence embeddings.

### 4. SIF Weighted Average of GloVe Embeddings
We followed the same procedure as outlined in [4]:

![algorithm](https://github.com/user-attachments/assets/4fb53a0c-745e-4ac2-8eae-f9160497a1d3)

### 5. Sent2Vec Embeddings
- Sent2vec is trained on sentences using vectors such as wordn-grams. 
- That is, in the following stages: 

- Split sentences into words, create n-grams vectors for words, calculate the average of the vectors, then train the model 

- In this method, we used a pre-trained model for sent2vec which contains one million and fifty thousand words 
and each word is represented by 600 vectors, then we calculated the sentence embeddings then, we calculated the distance between each of the two sentences mentioned above features

- In the Pagliardinietla.2018 paper, we followed the Mikolovetla.2013 approach, but at the sentence level.

### 6. Doc2Vec Embeddings
- In this method, we downloaded the Doc2vec library and trained it using words from the Brown Corpus, which is a small set of texts compared to the data, as the Brown Corpus contains 57 thousand non-words.

- Duplicate, it was chosen due to time constraints and modest capabilities.

- After training, the Doc2vec model stores words and their vectors.

- Then we calculated the sentence embeddings for each sentence in the form of tokens, and then we calculated the features between

- All two sentences mentioned previously. This idea is taken from Quoc et la 2014, where they followed the approach of 2013. Mikolov et la, where they designed a network similar to the CBOW network and for each of them they added a paragraph id, which represents the paragraph.

- As a whole:
  
![algorithm](https://github.com/user-attachments/assets/9bda26b5-f042-4f00-8757-c12bc996362d)
- They followed the skip-gram method, but the desired output is paragraph id.
  
![algorithm](https://github.com/user-attachments/assets/d28fcf9d-4cc9-4392-b960-15709cff53ac)
- Following these two methods results in word embeddings that carry the context information contained within it Best sentence embeddings.
  
## Setup
- requirements [here](https://github.com/Haidar-Al-Sous/Semantic-matching/blob/main/requirements.txt)
- Install the requirements using :
```bash
pip install -r requirements.txt
```
## Comparison table
| Model Name                                   | Train Log Loss | Test Log Loss |
|----------------------------------------------|----------------|---------------|
| Unweighted Average of Word2vec Embeddings    | 12.32          | 7.55          |
| Unweighted Average of Siamese CBOW Embeddings| 13.45          | 8.07          |
| Sentence-BERT Embeddings                     | 8.49           | 7.10          |
| SIF Weighted Average of Glove Embeddings     | 11.84          | 8.03          |
| Sent2vec Embeddings                          | 15.02          | 11.03         |
| Doc2vec Embeddings                           | 13.53          | 9.47          |

## References
[1]    Kenter, T., Borisov, A., & De Rijke, M. (2016). Siamese CBOW: Optimizing word embeddings for sentence representations. arXiv Preprint arXiv:1606.04640. https://doi.org/10.18653/v1/p16-1089  
[2]    Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. arXiv Preprint arXiv:1908.10084. https://doi.org/10.18653/v1/d19-1410  
[3]    Mikolov, T. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781  
[4]    Arora, S., Liang, Y., & Ma, T. (2017). A simple but Tough-to-Beat baseline for sentence embeddings. International Conference on Learning Representations. https://oar.princeton.edu/bitstream/88435/pr1rk2k/1/BaselineSentenceEmbedding.pdf  
[5]    Pagliardini, M., Gupta, P., & Jaggi, M. (2017). Unsupervised learning of sentence embeddings using compositional n-gram features. arXiv preprint arXiv:1703.02507  
[6]    Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents. International Conference on Machine Learning, 4, 1188–1196. http://ece.duke.edu/~lcarin/ChunyuanLi4.17.2015.pdf  

## Contributors
- [Ahmad-AM0](https://github.com/Ahmad-AM0)
- [Haidar-Al-Sous](https://github.com/Haidar-Al-Sous)
- [kenan-azd-dev](https://github.com/kenan-azd-dev)
- marianne deep
- hrayr derbedrossian 
