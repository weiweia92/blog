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

![](https://latex.codecogs.com/png.image?\dpi{110}%20\bigtriangledown_{\theta}J_t(\theta))将会非常稀疏，所以我们可能只更新实际出现的向量

解决方案：

* 需要稀疏矩阵更新操作只更新矩阵U和V中的特定行
* 需要保留单词向量的散列

如果数百万个单词向量并且进行分布式计算，那么重要的是不必到处发送巨大的更新

### Word2vec:More detail

为什么两个向量？     
* 更容易优化，最后取平均值     
* 可以每个单词只用一个向量     

两个模型变体：

* Skip-grams(SG):输入中心词并预测上下文中的单词     
* Continuous Bag of Words(CBOW):输入上下文的单词并预测中心词

之前一直使用naive的softmax(简单但代价很高的训练方法)，接下来使用负采样方法加快训练速率 

### The skip-gram model with negative sampling (HW2)

softmax中用于归一化的分母的计算代价太高,我们将在作业2中实现使用 negative sampling 负采样方法的 skip-gram 模型:使用一个 true pair (中心词及其上下文窗口中的词)与几个 noise pair (中心词与随机词搭配) 形成的样本，训练二元逻辑回归.原文中的(最大化)目标函数是

![](https://latex.codecogs.com/png.image?\dpi{110}%20J(\theta)=\frac{1}{T}\sum_{t=1}^{T}J_t(\theta))  
![]()



