# Comparative Study of Semantic Matching Techniques

## Introduction

We evaluated different embeddings techniques using this [dataset](https://www.kaggle.com/competitions/quora-question-pairs) in the context of semantic matching. We used six techniques:
1. **Unweighted Average of Word2Vec Embeddings.**
2. **Unweighted Average of Siamese CBOW Embeddings.**
3. **Sentence-BERT Embeddings.**
4. **SIF Weighted Average of GloVe Embeddings.**
5. **Sent2Vec Embeddings.**
6. **Doc2Vec Embeddings.**

Before explaining each appproach, we want to emphasize that the goal is to calculate sentence embeddings in the most accurate way.

### 1. Unweighted Average of Word2Vec Embeddings
  This approach consists of simply taking average of words' embeddings (using Word2Vec) for each sentence.

### 2. Unweighted Average of Siamese CBOW Embeddings
  In this approach, the goal is to compute words embeddings in a way that use info about the nearby sentences. First, We initialize embeddings for each word randomly. Then, we take the average of them for each sentence. After that, 

## Setup
- requirements [here](https://github.com/Haidar-Al-Sous/Semantic-matching/blob/main/requirements.txt)
- Install the requirements using :
```bash
pip install -r requirements.txt
```
## Results
-
-

## References
- Kenter, T., Borisov, A., & De Rijke, M. (2016). Siamese CBOW: Optimizing word embeddings for sentence representations. arXiv Preprint arXiv:1606.04640. https://doi.org/10.18653/v1/p16-1089
- Arora, S., Liang, Y., & Ma, T. (2017). A simple but Tough-to-Beat baseline for sentence embeddings. International Conference on Learning Representations. https://oar.princeton.edu/bitstream/88435/pr1rk2k/1/BaselineSentenceEmbedding.pdf
- Mikolov, T. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
- Pagliardini, M., Gupta, P., & Jaggi, M. (2017). Unsupervised learning of sentence embeddings using compositional n-gram features. arXiv preprint arXiv:1703.02507.
- Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents. International Conference on Machine Learning, 4, 1188–1196. http://ece.duke.edu/~lcarin/ChunyuanLi4.17.2015.pdf
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. arXiv Preprint arXiv:1908.10084. https://doi.org/10.18653/v1/d19-1410

## Contributors
- [Ahmad-AM0](https://github.com/Ahmad-AM0)
- [Haidar-Al-Sous](https://github.com/Haidar-Al-Sous)
- [kenan-azd-dev](https://github.com/kenan-azd-dev)
- marianne deep
- hrayr derbedrossian 
