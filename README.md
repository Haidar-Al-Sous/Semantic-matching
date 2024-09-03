# Comparative Study of Semantic Matching Techniques

## Introduction
We aim to compare different semantic matching techniques in this study. The goal is to calculate sentence embeddings in the most accurate manner. We selected six approaches and compared them using a [dataset](https://www.kaggle.com/competitions/quora-question-pairs) from Kaggle, which consists of two sentences in each sample along with their similarity. Our approach consists of two stages:
1. Computing features.
2. Training a logistic regression model.

For each pair of sentences, we computed three features:
1. Vector of absolute difference between the sentence embeddings.
2. Cosine similarity.
3. Euclidean distance.

Then, we trained a logistic regression model based on the computed features.

We chose the following approaches:
1. **Unweighted Average of Word2Vec Embeddings.**
2. **Unweighted Average of Siamese CBOW Embeddings.**
3. **Sentence-BERT Embeddings.**
4. **SIF Weighted Average of GloVe Embeddings.**
5. **Sent2Vec Embeddings.**
6. **Doc2Vec Embeddings.**

## Method
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
We used `FastSentenceTransformer` to compute sentence embeddings. You can refer to [2] for more information.

### 4. SIF Weighted Average of GloVe Embeddings
We followed the same procedure as outlined in [3]:  

![algorithm](https://github.com/user-attachments/assets/4fb53a0c-745e-4ac2-8eae-f9160497a1d3)  
We used glove embeddings from [here](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) and word frequencies from [here](https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt).

### 5. Sent2Vec Embeddings [4]
We used `sent2vec` library and `wiki_unigram` pretrained model. This appraoch is based heavily on continous bag-of-words and skip-gram models but on sentence level.

### 6. Doc2Vec Embeddings [5]
We used the `Gensim` library to compute document embeddings after training it on the Brown corpus. Doc2Vec approach rests on the same ideas used in Word2Vec, namely, continous bag-of-words and skip-gram models.

Distributed Memory Model of Paragraph Vectors (derived from continous bag-of-words model):

![PV-DM](https://github.com/user-attachments/assets/a2a9089e-f274-4def-af06-c4ec6d020363)  

Distributed Bag of Words of Paragraph Vector (derived from  skip-gram model):

![PV-DBOW](https://github.com/user-attachments/assets/3ba3856b-e918-48e5-8e6f-2feee4731841)  

## Results
We compared the results we obtained on Kaggle and found that Sentence-BERT Embeddings surpassed all of them.
| Model Name                                   | Train Log Loss | Test Log Loss |
| :---                                         |     :---:      |     :---:     |
| Unweighted Average of Word2vec Embeddings    | 12.32          | 7.55          |
| Unweighted Average of Siamese CBOW Embeddings| 13.45          | 8.07          |
| Sentence-BERT Embeddings                     | 8.49           | 7.10          |
| SIF Weighted Average of Glove Embeddings     | 11.84          | 8.03          |
| Sent2vec Embeddings                          | 15.02          | 11.03         |
| Doc2vec Embeddings                           | 13.53          | 9.47          |

## References
[1]    Kenter, T., Borisov, A., & De Rijke, M. (2016). Siamese CBOW: Optimizing word embeddings for sentence representations. arXiv Preprint arXiv:1606.04640. https://doi.org/10.18653/v1/p16-1089.  
[2]    Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. arXiv Preprint arXiv:1908.10084. https://doi.org/10.18653/v1/d19-1410.  
[3]    Arora, S., Liang, Y., & Ma, T. (2017). A simple but Tough-to-Beat baseline for sentence embeddings. International Conference on Learning Representations.  
[4]    Pagliardini, M., Gupta, P., & Jaggi, M. (2017). Unsupervised learning of sentence embeddings using compositional n-gram features. arXiv preprint arXiv:1703.02507.  
[5]    Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents. International conference on machine learning (pp. 1188-1196). PMLR.  

## Contributors
- [Ahmad-AM0](https://github.com/Ahmad-AM0)
- [Haidar-Al-Sous](https://github.com/Haidar-Al-Sous)
- [kenan-azd-dev](https://github.com/kenan-azd-dev)
- [mariannedeeb](https://github.com/mariannedeeb/)
- [hrayrdb](https://github.com/hrayrdb/)
