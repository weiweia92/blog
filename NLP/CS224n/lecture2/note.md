# Lecture 02 Word Vectors and Word Senses

## Lecture Plan

* Finish looking at word vectors and word2vec

* Optimization basics

* Can we capture this essence more effectively by counting?

* The GloVe model of word vectors

* Evaluating word vectors

* Word senses

* The course

Goal: be able to read word embeddings papers by the end of class.

**Review: Main idea of word2vec**

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.52.29%20AM.png)

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(o|c)=\frac{exp(u_o^Tv_c)}{\sum_{w%20\in%20V}exp(u_w^Tv_c)})

* 遍历整个语料库中的每个单词

* 使用单词向量预测周围的单词

* 更新向量以便更好地预测

### Word2vec parameters and computations

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%209.01.08%20AM.png)     
![](https://latex.codecogs.com/png.image?\dpi{110}%20(n%20\times%20d)\cdot(d\times%201)\rightarrow%20(n\times%201)\overset{softmax}{\rightarrow}(n\times%201))

* 每行代表一个单词的词向量，点乘后得到的分数通过softmax映射为概率分布，并且我们得到的概率分布是对于该中心词而言的上下文中单词的概率分布，该分布于上下文所在的具体位置无关，所以每个位置的预测都一样

* 我们希望模型对上下文中(相对频繁)出现的所有单词给出个合理的高概率估计

* the, and ,that, of这样的停用词，是每个单词点乘后得到的较大概率的单词
   * 去掉这一部分可以使词向量效果更好

## Optimization: Gradient Descent

Gradient Descent 每次使用全部样本进行更新

Stochastic Gradient Descent 每次只是用单个样本进行更新

Mini-batch具有以下优点     
* 通过平均值，减少梯度估计的噪音

* 在GPU上并行化运算，加快运算速度

### Stochastic gradients with word vectors

![](https://latex.codecogs.com/png.image?\dpi{110}%20\bigtriangledown_{\theta}J_t(\theta))
