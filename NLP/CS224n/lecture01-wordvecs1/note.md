# Lecture01 structure
* The course

* Human language and word meaning

* Word2vec introduction

* Word2vec objective function gradients

* Optimization basics

* Looking at word vectors    

## Human language and word meaning

人类之所以比类人猿更“聪明”，是因为我们有语言，因此是一个人机网络，其中人类语言作为网络语言。人类语言具有 信息功能 和 社会功能 。  
据估计，人类语言只有大约5000年的短暂历。语言是人类变得强大的主要原因。写作是另一件让人类变得强大的事情。它是使知识能够在空间上传送到世界各地，并在时间上传送的一种工具。  
但是，相较于如今的互联网的传播速度而言，人类语言是一种缓慢的语言。然而，只需人类语言形式的几百位信息，就可以构建整个视觉场景。这就是自然语言如此迷人的原因。   

### How do we represent the meaning of a word?   

**meaning**     

* 用一个词、词组等表示的概念。

* 一个人想用语言、符号等来表达的想法。

* 表达在作品、艺术等方面的思想    

理解意义的最普遍的语言方式(linguistic way) : 语言符号与语言符号的意义的转化     

![](https://latex.codecogs.com/png.image?\dpi{110}%20signifier(symbol)\Leftrightarrow%20signified(idea%20\%20or%20\%20thing)=denotational%20\%20semantics)     

**denotational semantics指称语义**       

### How do we have usable meaning in a computer?**          

WordNet, 一个包含同义词集和上位词(“is a”关系) synonym sets and hypernyms 的列表的辞典    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.27.30%20AM.png)       

**Problems with resources like WordNet**   

* 作为一个资源很好，但忽略了细微差别    
   * 例如“proficient”被列为“good”的同义词。这只在某些上下文中是正确的。
* 缺少单词的新含义

* 难以持续更新
   * 例如 wicked, badass, nifty, wizard, genius, ninja, bombest

* 主观的

* 需要人类劳动来创造和调整

* 无法计算单词相似度   

### Representing words as discrete symbols

在传统的自然语言处理中，我们把词语看作离散的符号: hotel, conference, motel - a **localist** representation。单词可以通过独热向量(one-hot vectors，只有一个1，其余均为0的稀疏向量) 。向量维度=词汇量(如500,000)。  

![](https://latex.codecogs.com/png.image?\dpi{110}%20motel=[0%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%201%20\%200%20\%200%20\%200%20\%200%20\%200%20])   

![](https://latex.codecogs.com/png.image?\dpi{110}%20hotel=[0%20\%200%20\%200%20\%200%20\%200%20\%201%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20\%200%20])               

**Problem with words as discrete symbols**   

所有向量是正交的。对于独热向量，没有关于相似性概念，并且向量维度过大。    

**Solutions**    

* 使用类似 WordNet 的工具中的列表，获得相似度，但会因不够完整而失败

* **Instead:learn to encode similary in the vectors themselves**      

### Representing words by their context

* Distributional semantics ：一个单词的意思是由经常出现在它附近的单词给出的    
   * “You shall know a word by the company it keeps” (J. R. Firth 1957: 11)     
   * 现代统计NLP最成功的理念之一     
   * 有点物以类聚，人以群分的感觉

* 当一个单词![](https://latex.codecogs.com/png.image?\dpi{110}%20w)出现在文本中时，它的上下文是出现在其附近的一组单词(在一个固定大小的窗口中)。    

* 使用![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的许多上下文来构建![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的表示.    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.40.39%20AM.png)     

## Word2vec introduction

我们为每个单词构建一个 密集(dense)的向量，使其与出现在相似上下文中的单词向量相似.  

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.44.27%20AM.png)    

词向量 word vectors 有时被称为词嵌入 **word embeddings** 或词表示 word representations.     

它们是分布式表示 **distributed representation**.    

**Word2vec (Mikolov et al. 2013)是一个学习单词向量的framework**     

Idea:     

* We have a large corpus (“body”) of text     

* Every word in a fixed vocabulary is represented by a vector     

* Go through each position ![](https://latex.codecogs.com/png.image?\dpi{110}%20t) in the text, which has a center word ![](https://latex.codecogs.com/png.image?\dpi{110}%20c) and context(“outside”) words ![](https://latex.codecogs.com/png.image?\dpi{110}%20o)     

* Use the similarity of the word vectors for ![](https://latex.codecogs.com/png.image?\dpi{110}%20c) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20o) to calculate the probability of ![](https://latex.codecogs.com/png.image?\dpi{110}%20o) given ![](https://latex.codecogs.com/png.image?\dpi{110}%20c) (or vice versa)     

* Keep adjusting the word vectors to maximize this probability     

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.52.29%20AM.png)    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%209.52.38%20AM.png)    

## Word2vec objection function

对于每个位置![](https://latex.codecogs.com/png.image?\dpi{110}%20t=1,...T)，在大小为![](https://latex.codecogs.com/png.image?\dpi{110}%20m)的固定窗口内预测上下文单词，给定中心词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_j)    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%2010.01.11%20AM.png)    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%2010.01.37%20AM.png)    

公式中，向量![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o)和向量![](https://latex.codecogs.com/png.image?\dpi{110}%20v_o)进行点乘。向量之间越相似点乘结果越大，从而归一化后得到的概率值也越大。   

**模型的训练正是为了使得具有相似上下文的单词，拥有相似的向量。**     

## Word2vec prediction function

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(o|c)=\frac{exp(u_o^T%20v_c)}{\sum_{w\in%20V}exp(u_w^T%20v_c)})   

* 取幂使任何数都为正    

* 点积比较![](https://latex.codecogs.com/png.image?\dpi{110}%20o)和![](https://latex.codecogs.com/png.image?\dpi{110}%20c)的相似性，![](https://latex.codecogs.com/png.image?\dpi{110}%20u^T%20v=u%20\cdot%20v=\sum_{i=1}^{n}u_iv_i)，点积越大则概率越大。    

* 分母：对整个词汇表进行标准化，从而得到概率分布    

首先我们**随机初始化**![](https://latex.codecogs.com/png.image?\dpi{110}%20u_w%20\in%20\mathbb{R}^d)和![](https://latex.codecogs.com/png.image?\dpi{110}%20v_w%20\in%20\mathbb{R}^d)，而后使用梯度下降法进行更新.    

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20}{\partial%20v_c}logP(o|c)=\frac{\partial%20}{\partial%20v_c}log\frac{exp(u_o^Tv_c)}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}=\frac{\partial%20}{\partial%20v_c}(log\%20exp(u_o^Tv_c)%20-log{\sum_{w%20\in%20V}exp(u_w^Tv_c)}))      
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\frac{\partial%20}{\partial%20v_c}(u_o^T%20v_c-log\sum_{w%20\in%20V}exp(u_w^T%20v_c)))    
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o-\frac{1}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}\cdot%20\frac{\partial%20}{\partial%20v_c}\sum_{w%20\in%20V}exp(u_w^Tv_c))      
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o-\frac{1}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}\cdot%20\sum_{w%20\in%20V}exp(u_w^Tv_c)\cdot%20\frac{\partial%20}{\partial%20v_c}u_w^Tv_c)     
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o-\sum_{w%20\in%20V}\frac{exp(u_w^Tv_c)}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}\cdot%20u_w)      
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o-\sum_{w%20\in%20V}P(w|c)u_w)      

再对![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o)进行偏微分计算，注意这里的![](![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o))是![](https://latex.codecogs.com/png.image?\dpi{110}%20u_{w=o})的简写，故可知    

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20}{\partial%20u_o}\sum_{w%20\in%20V}u_w^T%20v_c=\frac{\partial%20}{\partial%20u_o}\sum_{w%20\in%20V}u_o^T%20v_c=v_c)     

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20}{\partial%20u_o}logP(o|c)=\frac{\partial%20}{\partial%20u_o}log\frac{exp(u_o^Tv_c)}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}=\frac{\partial%20}{\partial%20u_o}(log\%20exp(u_o^Tv_c)-log\sum_{w%20\in%20V}exp(u_w^Tv_c)))   
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\frac{\partial%20}{\partial%20u_o}(u_o^Tv_c-log\sum_{w%20\in%20V}exp(u_w^T%20v_c)))     
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=v_o-\frac{1}{\sum_{w%20\in%20V}exp(u_w^T%20v_c)}\cdot%20\sum_{w%20\in%20V}\frac{\partial%20}{\partial%20u_o}exp(u_w^Tv_c))     
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=v_o-\frac{1}{\sum_{w%20\in%20V}exp(u_w^T%20v_c)}\cdot%20\sum_{w%20\in%20V}exp(u_w^Tv_c)\frac{\partial%20}{\partial%20u_o}%20\cdot%20u_w^Tv_c)     
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=v_c-\frac{exp(u_o^Tv_c)}{\sum_{w%20\in%20V}exp(u_w^Tv_c)}\cdot%20v_c)   
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=v_c-P(o|c)\cdot%20v_c=(1-P(o|c))v_c)     

可以理解，当![](https://latex.codecogs.com/png.image?\dpi{110}%20P(o|c)\to%201)，即通过中心词![](https://latex.codecogs.com/png.image?\dpi{110}%20c)我们可以正确预测上下文词![](https://latex.codecogs.com/png.image?\dpi{110}%20o)，此时我们不需要调整![](https://latex.codecogs.com/png.image?\dpi{110}%20o)，反之，则需要调整![](https://latex.codecogs.com/png.image?\dpi{110}%20o)。    

# Notes01 Introduction SVD and Word2vec

## Introduction to Natural Language Processing

**What is so special about NLP?**    

Natural language is a discrete/symbolic/categorical system.   

离散的/符号的/分类的     

人类的语言有什么特别之处？人类语言是一个专门用来表达意义的系统，而不是由任何形式的物理表现产生的。在这方面上，它与视觉或任何其他机器学习任务都有很大的不同。     

大多数单词只是一个语言学以外的的符号：单词是一个映射到所指(signified 想法或事物)的能指(signifier)。      

例如，“rocket”一词指的是火箭的概念，因此可以引申为火箭的实例。当我们使用单词和字母来表达符号时，也会有一些例外，例如“whoompaa”的使用。最重要的是，这些语言的符号可以被 编码成几种形式：声音、手势、文字等等，然后通过连续的信号传输给大脑，大脑本身似乎也能以一种连续的方式对这些信号进行解码。人们在语言哲学和语言学方面做了大量的工作来概念化人类语言，并将词语与其参照、意义等区分开来。    

Examples of tasks

自然语言处理有不同层次的任务，从语言处理到语义解释再到语篇处理。自然语言处理的目标是通过设计算法使得计算机能够“理解”语言，从而能够执行某些特定的任务。不同的任务的难度是不一样的

Easy  

* 拼写检查 Spell Checking

* 关键词检索 Keyword Search

* 同义词查找 Finding Synonyms     

Medium    

* 解析来自网站、文档等的信息

Hard    

* 机器翻译 Machine Translation

* 语义分析 Semantic Analysis

* 指代消解 Coreference

* 问答系统 Question Answering

**How to represent words?**      

在所有的NLP任务中，第一个也是可以说是最重要的共同点是我们如何将单词表示为任何模型的输入。在这里我们不会讨论早期的自然语言处理工作是将单词视为原子符号 atomic symbols。为了让大多数的自然语言处理任务能有更好的表现，我们首先需要了解单词之间的相似和不同。有了词向量，我们可以很容易地将其编码到向量本身中。

## Word Vectors

使用词向量编码单词，N维空间足够我们编码语言的所有语义，每一维度都会编码一些我们使用语言传递的信息。简单的one-hot向量无法给出单词间的相似性，我们需要将维度![](https://latex.codecogs.com/png.image?\dpi{110}%20|V|)减少至一个低纬度的子空间，来获得稠密的词向量，获得词之间的关系。

## SVD Based Methods

这是一类找到词嵌入的方法（即词向量），我们首先遍历一个很大的数据集和统计词的共现计数矩阵(co-occurrence) X，然后对矩阵 X 进行 SVD 分解得到![](https://latex.codecogs.com/png.image?\dpi{110}%20USV^T) 。然后我们使用 U 的行来作为字典中所有词的词向量。我们来讨论一下矩阵 X 的几种选择。

### Word-Document Matrix

我们最初的尝试，我们猜想相关连的单词在同一个文档中会经常出现。例如，“banks” “bonds” “stocks” “moneys”等等，出现在一起的概率会比较高。但是“banks” “octopus” “banana” “hockey”不大可能会连续地出现。我们根据这个情况来建立一个 Word-Document 矩阵，![](https://latex.codecogs.com/png.image?\dpi{110}%20X)是按照以下方式构建：遍历数亿的文档和当词 

![](https://latex.codecogs.com/png.image?\dpi{110}%20i)出现在文档![](https://latex.codecogs.com/png.image?\dpi{110}%20j)时，我们对 ![](https://latex.codecogs.com/png.image?\dpi{110}%20X_{ij})加一。这显然是一个很大的矩阵 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\mathbb{R}^{|V|\times%20M})，它的规模是和文档数量![](https://latex.codecogs.com/png.image?\dpi{110}%20M)成正比关系。因此我们可以尝试更好的方法。

### Window based Co-occurrence Matrix

同样的逻辑也适用于这里，但是矩阵 ![](https://latex.codecogs.com/png.image?\dpi{110}%20X)存储单词的共现，从而成为一个关联矩阵。在此方法中，我们计算每个单词在特定大小的窗口中出现的次数。我们按照这个方法对语料库中的所有单词进行统计。    

* 生成维度为![](https://latex.codecogs.com/png.image?\dpi{110}%20|V|\times%20|V|)的共现矩阵![](https://latex.codecogs.com/png.image?\dpi{110}%20X)

* 在![](https://latex.codecogs.com/png.image?\dpi{110}%20X)上应用SVD得到![](https://latex.codecogs.com/png.image?\dpi{110}%20X=USV^T)

* 选择前![](https://latex.codecogs.com/png.image?\dpi{110}%20k)行得到![](https://latex.codecogs.com/png.image?\dpi{110}%20k)维的词向量

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\sum_{i=1}^{k}\sigma_i}{\sum_{i=1}^{|V|}\sigma%20_i})表示第一个![](https://latex.codecogs.com/png.image?\dpi{110}%20k)维捕获的方差量     

### Applying SVD to the cooccurrence matrix

我们对矩阵  ![](https://latex.codecogs.com/png.image?\dpi{110}%20X)使用 SVD，观察奇异值（矩阵 S 上对角线上元素），根据期望的捕获方差百分比截断，留下前![](https://latex.codecogs.com/png.image?\dpi{110}%20k)个元素：     

然后取子矩阵 ![](https://latex.codecogs.com/png.image?\dpi{110}%20U_{1:V,1:k})作为词嵌入矩阵。这就给出了词汇表中每个词的 ![](https://latex.codecogs.com/png.image?\dpi{110}%20k)维表示

对矩阵![](https://latex.codecogs.com/png.image?\dpi{110}%20X)使用SVD        

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%2011.44.26%20AM.png)  

通过选择前![](https://latex.codecogs.com/png.image?\dpi{110}%20k)个奇异向量来降低维度   

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%2011.44.36%20AM.png)    

这两种方法都给我们提供了足够的词向量来编码语义和句法(part of speech)信息，但伴随许多其他问题

* 矩阵的维度会经常发生改变（经常增加新的单词和语料库的大小会改变）。

* 矩阵会非常的稀疏，因为很多词不会共现。

* 矩阵维度一般会非常高 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\approx%2010^6%20\times%2010^6)

* 基于 SVD 的方法的计算复杂度很高(![](https://latex.codecogs.com/png.image?\dpi{110}%20m%20\times%20n))矩阵的计算成本是(![](https://latex.codecogs.com/png.image?\dpi{110}%20O(mn^2)))，并且很难合并新单词或文档

* 需要在 X 上加入一些技巧处理来解决词频的极剧的不平衡

然而，基于计数的方法可以有效地利用统计量

对上述讨论中存在的问题存在以下的解决方法：

* 忽略功能词，例如 “the”，“he”，“has” 等等。

* 使用 ramp window，即根据文档中单词之间的距离对共现计数进行加权

* 使用皮尔逊相关系数并将负计数设置为0，而不是只使用原始计数

正如我们在下一节中看到的，基于迭代的方法以一种优雅得多的方式解决了大部分上述问题。

## Iteration Based Methods - Word2vec

这里我们尝试一个新的方法。我们可以尝试创建一个模型，该模型能够一次学习一个迭代，并最终能够对给定上下文的单词的概率进行编码，而不是计算和存储一些大型数据集(可能是数十亿个句子)的全局信息.

这个想法是设计一个模型，该模型的参数就是词向量。然后根据一个目标函数训练模型，在每次模型的迭代计算误差，并遵循一些更新规则，该规则具有惩罚造成错误的模型参数的作用，从而可以学习到词向量。这个方法可以追溯到 1986年，我们称这个方法为“反向传播”，模型和任务越简单，训练它的速度就越快。

* 基于迭代的方法一次捕获一个单词的共现情况，而不是像SVD方法那样直接捕获所有的共现计数。

已经很多人按照这个思路测试了不同的方法。[Collobert et al., 2011] 设计的模型首先将每个单词转换为向量。对每个特定的任务（命名实体识别、词性标注等等），他们不仅训练模型的参数，同时也训练单词向量，计算出了非常好的词向量的同时取得了很好的性能。

在这里，我们介绍一个非常有效的概率模型：Word2vec。Word2vec 是一个软件包实际上包含：

* 两个算法：continuous bag-of-words（CBOW）和 skip-gram。CBOW 是根据中心词周围的上下文单词来预测该词的词向量。skip-gram 则相反，是根据中心词预测周围上下文的词的概率分布。

* 两个训练方法：negative sampling 和 hierarchical softmax。Negative sampling 通过抽取负样本来定义目标，hierarchical softmax 通过使用一个有效的树结构来计算所有词的概率来定义目标。  

### Language Models(Unigrams, Bigrams, etc)

首先，我们需要创建一个模型来为一系列的单词分配概率。我们从一个例子开始：

“The cat jumped over the puddle”

一个好的语言模型会给这个句子很高的概率，因为在句法和语义上这是一个完全有效的句子。相似地，句子“stock boil fish is toy”会得到一个很低的概率，因为这是一个无意义的句子。在数学上，我们可以称为对给定 n 个词的序列的概率是：

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(w_1,%20w_2,%20\cdots%20,w_n))

我们可以采用一元语言模型方法(Unigram model)，假设单词的出现是完全独立的，从而分解概率

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(w_1,%20w_2,%20\cdots%20,w_n)=\prod%20_{i=1}^{n}P(w_i))

但是我们知道这是不大合理的，因为下一个单词是高度依赖于前面的单词序列的。如果使用上述的语言模型，可能会让一个无意义的句子具有很高的概率。所以我们让序列的概率取决于序列中的单词和其旁边的单词的成对概率。我们称之为 bigram 模型：

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(w_1,%20w_2,%20\cdots%20,w_n)=\prod%20_{i=1}^{n}P(w_i|w_{i-1}))

但是，这个方法还是有点简单，因为我们只关心一对邻近的单词，而不是针对整个句子来考虑。但是我们将看到，这个方法会有显著的提升。考虑在词-词共现矩阵中，共现窗口为 1，我们基本上能得到这样的成对的概率。但是，这又需要计算和存储大量数据集的全局信息。

既然我们已经理解了如何考虑具有概率的单词序列，那么让我们观察一些能够学习这些概率的示例模型。

### Continuous Bag of Words Model(CBOW)

这一方法是把 {"The","cat","over","the","puddle"} 作为上下文，希望从这些词中能够预测或者生成中心词“jumped”。这样的模型我们称之为 continuous bag-of-words（CBOW）模型。

首先我们设定已知参数。令我们模型的已知参数是 one-hot 形式的词向量表示。输入的 one-hot 向量或者上下文我们用 ![](https://latex.codecogs.com/png.image?\dpi{110}%20x_c)表示，输出用 

![](https://latex.codecogs.com/png.image?\dpi{110}%20y_c)表示。在 CBOW 模型中，因为我们只有一个输出，因此我们把 y 称为是已知中心词的的 one-hot 向量。现在让我们定义模型的未知参数。

首先我们对 CBOW 模型作出以下定义.  

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)：词汇表![](https://latex.codecogs.com/png.image?\dpi{110}%20V)中的单词![](https://latex.codecogs.com/png.image?\dpi{110}%20i)

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu%20\in%20\mathbb{R}^{n%20\times%20|V|})：输入词矩阵

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20\upsilon_i)：![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu)的第![](https://latex.codecogs.com/png.image?\dpi{110}%20i)列，单词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)的输入向量表示

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20u%20\in%20\mathbb{R}^{|V|\times%20n})：输出词矩阵

* ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_i)：![](https://latex.codecogs.com/png.image?\dpi{110}%20u)的第![](https://latex.codecogs.com/png.image?\dpi{110}%20i)行，单词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)的输出向量表示

我们创建两个矩阵，![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu%20\in%20\mathbb{R}^{n%20\times%20|V|})和![](https://latex.codecogs.com/png.image?\dpi{110}%20u%20\in%20\mathbb{R}^{|V|\times%20n})。其中n是嵌入空间的任一维度大小。![](https://latex.codecogs.com/png.image?\dpi{110}%20V)是输入词矩阵，

使得当其为模型的输入时，![](![](https://latex.codecogs.com/png.image?\dpi{110}%20V))的第i列是词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)的n维嵌入向量。我们定义![](https://latex.codecogs.com/png.image?\dpi{110}%20n\times%201)的向量为![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu_i)。

相似的，![](https://latex.codecogs.com/png.image?\dpi{110}%20u)是输出矩阵。当其为模型的输入时，![](https://latex.codecogs.com/png.image?\dpi{110}%20u)的第j行是词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_j)的n维嵌入向量。我们定义![](https://latex.codecogs.com/png.image?\dpi{110}%20u)的这行为

![](https://latex.codecogs.com/png.image?\dpi{110}%20u_j)。注意：实际上对每个词![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)我们需要学习两个词向量（即输入词向量![](https://latex.codecogs.com/png.image?\dpi{110}%20v_i)和输出词向量![](https://latex.codecogs.com/png.image?\dpi{110}%20u_i)）.

我们将这个模型分解为以下步骤

* 我们为大小为 m 的输入上下文，生成 one-hot 词向量      
  
  ![](https://latex.codecogs.com/png.image?\dpi{110}%20(x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x^{(c+m)}\in%20\mathbb{R}^{|V|}))

* 我们从上下文![](https://latex.codecogs.com/png.image?\dpi{110}%20(v_{(c-m)}=\nu%20x^{(c-m)},...,v_{(c-m+1)}=\nu%20x^{(c-m+1)},...,v_{(c+m)}=\nu%20x^{(c+m)}\in%20\mathbb{R}^{n}))得到嵌入词向量

* 对上述的向量求平均值![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}%20\in%20\mathbb{R}^{n})

* 生成一个分数向量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20z=u\hat{v}%20\in%20\mathbb{R^{|V|}})。当相似向量的点积越高，就会令相似的词更靠近，从而获得更高的分数。将分数转换为概率![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}=softmax(z)%20\in%20\mathbb{R}^{|V|})
   * 这里 softmax 是一个常用的函数。它将一个向量转换为另外一个向量，其中转换后的向量的第i个元素是![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{e^{\hat{y_i}}}{\sum_{k=1}^{|V|}e^{\hat{y_k}}})。
     因为该函数是一个指数函数，所以值一定为正数；通过除以![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_{k=1}^{|V|}e^{\hat{y_k}})来进行归一化向量，使得![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_{k=1}^{|V|}{\hat{y_k}}=1)得到概率。

* 我们希望生成的概率![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}%20\in%20\mathbb{R}^{|V|})与实际的概率![](https://latex.codecogs.com/png.image?\dpi{110}%20y%20\in%20\mathbb{R}^{|V|})匹配。使得其刚好是实际的词就是这个 one-hot 向量。

下图是 CBOW 模型的计算图示    

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%202.52.11%20PM.png)

如果有![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu)和![](https://latex.codecogs.com/png.image?\dpi{110}%20u)，我们知道这个模型是如何工作的，那我们如何学习这两个矩阵呢？这需要创建一个目标函数。一般我们想从一些真实的概率中学习一个概率，信息论提供了一个**度量两个概率分布的距离** 的方法。这里我们采用一个常见的距离/损失方法，交叉熵 ![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},%20y))
 
在离散情况下使用交叉熵可以直观地得出损失函数的公式

![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},%20y)=-\sum_{j=1}^{|V|}y_j%20log\hat{(y_j)})

上面的公式中，y 是 one-hot 向量。因此上面的损失函数可以简化为：

![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},%20y)=-y_j%20log\hat{(y_j)})

c 是正确词的 one-hot 向量的索引。我们现在可以考虑我们的预测是完美并且![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y_c}=1)的情况。然后我们可以计算![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},%20y)=-1log(1)=0)因此，对一个完美的预测，我们不会面临任何惩罚或者损失。现在我们考虑一个相反的情况，预测非常差并且![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y_c}=0.01)。
和前面类似，我们可以计算损失![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},%20y)=-1log(0.01)=4.605)。因此，我们可以看到，对于概率分布，交叉熵为我们提供了一个很好的距离度量。因此我们的优化目标函数公式为：

![](https://latex.codecogs.com/png.image?\dpi{110}%20minimize\%20J=-logP(w_c|w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m}))     
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-logP(u_c|\hat{v})=-log\frac{exp(u_c^T\hat{v})}{\sum_{j=1}^{|V|}exp(u_j^T\hat{v})})    
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-u_c^T\hat{v}+log\sum_{j=1}^{|V|}exp(u_j^T\hat{v}))    

我们使用 SGD 来更新所有相关的词向量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_c)和 ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_j)。SGD 对一个窗口计算梯度和更新参数

![](https://latex.codecogs.com/png.image?\dpi{110}%20u_{new}\leftarrow%20u_{old}-\alpha\bigtriangledown_{u}J)    
![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu_{new}\leftarrow%20\nu_{old}-\alpha\bigtriangledown_{v}J)     

### Skip-Gram Model

Skip-Gram模型与CBOW大体相同，但是交换了我们的 x 和 y，即 CBOW 中的 x 现在是 y，y 现在是 x。输入的 one-hot 向量（中心词）我们表示为 x，输出向量为![](https://latex.codecogs.com/png.image?\dpi{110}%20y^{j})。我们定义的![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu)
和![](https://latex.codecogs.com/png.image?\dpi{110}%20u)是和CBOW一样的

* 生成中心词的one-hot向量![](https://latex.codecogs.com/png.image?\dpi{110}%20x%20\in%20\mathbb{R}^{|V|})

* 我们对中心词![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c=\nu%20x\in%20\mathbb{R}^{|V|})得到词嵌入向量

* 生成分数向量![](https://latex.codecogs.com/png.image?\dpi{110}%20z%20=%20uv_c)

* 将分数向量转化为概率，![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}=softmax(z))注意![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y_{c-m}},...,\hat{y_{c-1}},\hat{y_{c+1}},...,\hat{y_{c+m}})是每个上下文词观察到的概率

* 我们希望我们生成的概率向量匹配真实概率![](https://latex.codecogs.com/png.image?\dpi{110}%20y^{(c-m)},...,y^{(c-1)},y^{(c+1)},...,y^{(c+m)})，one-hot向量是实际输出

下图是Skip-Gram模型的计算图示

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%203.34.59%20PM.png)

和 CBOW 模型一样，我们需要生成一个目标函数来评估这个模型。与 CBOW 模型的一个主要的不同是我们引用了一个朴素的贝叶斯假设来拆分概率。这是一个很强（朴素）的条件独立假设。换而言之，**给定中心词，所有输出的词是完全独立的**。(即公式1至2行)

![](https://latex.codecogs.com/png.image?\dpi{110}%20minimizeJ=-logP(w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m}|w_c))    
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-log\prod%20_{j=0,j\neq%20m}^{2m}P(w_{c-m+j}|w_c)=-log\prod%20_{j=0,j\neq%20m}^{2m}P(u_{c-m+j}|v_c))    
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-log\prod%20_{j=0,j\neq%20m}^{2m}\frac{exp(u_{c-m+j}^Tv_c)}{\sum_{k=1}^{|V|}exp(u_k^Tv_c)})   
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-\sum_{j=0,j\neq%20m}^{2m}logP(u_{c-m+j}|v_c))

通过这个目标函数，我们可以计算出与未知参数相关的梯度，并且在每次迭代中通过 SGD 来更新它们。

注意 

![](https://latex.codecogs.com/png.image?\dpi{110}%20J=-\sum_{j=0,j\neq%20m}^{2m}logP(u_{c-m+j}|v_c)=\sum_{j=0,j\neq%20m}^{2m}H(\hat{y},y_{c-m+j}))

其中![](https://latex.codecogs.com/png.image?\dpi{110}%20H(\hat{y},y_{c-m+j}))是向量![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y})的概率和one-hat向量![](https://latex.codecogs.com/png.image?\dpi{110}%20y_{c-m+j})之间的交叉熵。

### Negative Sampling

让我们再回到目标函数上。注意对![](https://latex.codecogs.com/png.image?\dpi{110}%20|V|)的求和计算量是非常大的。任何的更新或者对目标函数的评估都要花费![](https://latex.codecogs.com/png.image?\dpi{110}%20O(|V|))的时间复杂度。一个简单的想法是不去直接计算，而是去求近似值。

在每一个训练的时间步，我们不去遍历整个词汇表，而仅仅是抽取一些负样例。我们对噪声分布![](https://latex.codecogs.com/png.image?\dpi{110}%20P_n(w))“抽样”，这个概率是和词频的排序相匹配的。为加强对问题的表述以纳入负抽样，我们只需更新其

* 目标函数

* 梯度

* 更新规则 

Mikolov 在论文《Distributed Representations of Words and Phrases and their Compositionality.》中提出了负采样。虽然负采样是基于 Skip-Gram 模型，但实际上是对一个不同的目标函数进行优化。

考虑一对中心词和上下文词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20(w,%20c))。这词对是来自训练数据集吗？我们通过 ![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=1|w,c))表示 ![](https://latex.codecogs.com/png.image?\dpi{110}%20(w,%20c))是来自语料库。相应地， ![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=0|w,c))表示 ![](https://latex.codecogs.com/png.image?\dpi{110}%20(w,%20c))不是来自语料库。

首先，我们对 ![](![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=1|w,c)))用 sigmoid 函数建模：  

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=1|w,c,\theta)=\sigma(v_c^Tv_w)=\frac{1}{1+e^{(-v_c^Tv_w)}})   

现在，我们建立一个新的目标函数，如果中心词和上下文词确实在语料库中，就最大化概率![](![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=1|w,c))) ，如果中心词和上下文词确实不在语料库中，就最大化概率 
![](https://latex.codecogs.com/png.image?\dpi{110}%20P(D=0|w,c)) 。我们对这两个概率采用一个简单的极大似然估计的方法（这里我们把![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta)作为模型的参数，在我们的例子是 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\nu)和
 和 ![](https://latex.codecogs.com/png.image?\dpi{110}%20u)    
 
 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta=argmax\prod%20_{(w,c)\in%20D}P(D=1|w,c,\theta)\prod%20_{(w,c)\in%20\widetilde{D}}P(D=0|w,c,\theta))     
 &emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta=argmax\prod%20_{(w,c)\in%20D}P(D=1|w,c,\theta)\prod%20_{(w,c)\in%20\widetilde{D}}(1-P(D=1|w,c,\theta)))    
 &emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta=argmax\sum_{(w,c)\in%20D}logP(D=1|w,c,\theta)+\sum%20_{(w,c)\in%20\widetilde{D}}log(1-P(D=1|w,c,\theta)))     
 &emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta=argmax\sum_{(w,c)\in%20D}log\frac{1}{1+exp(-u_w^Tv_c)}+\sum%20_{(w,c)\in%20\widetilde{D}}log(1-\frac{1}{1+exp(-u_w^Tv_c)}))      
 &emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\theta=argmax\sum_{(w,c)\in%20D}log\frac{1}{1+exp(-u_w^Tv_c)}+\sum%20_{(w,c)\in%20\widetilde{D}}log(\frac{1}{1+exp(u_w^Tv_c)}))     
 
 注意最大化似然函数等同于最小化负对数似然：
 
 ![](https://latex.codecogs.com/png.image?\dpi{110}%20J=-\sum_{(w,c)\in%20D}log\frac{1}{1+exp(-u_w^Tv_c)}-\sum%20_{(w,c)\in%20\widetilde{D}}log(\frac{1}{1+exp(u_w^Tv_c)}))
 
 注意 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\widetilde{D})是“假的”或者“负的”语料。例如我们有句子类似“stock boil fish is toy”，这种无意义的句子出现时会得到一个很低的概率。我们可以从语料库中随机抽样出负样例 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\widetilde{D})。

对于 Skip-Gram 模型，我们对给定中心词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20c) 来观察的上下文单词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20c-m+j)的新目标函数为    

![](https://latex.codecogs.com/png.image?\dpi{110}%20-log\sigma(u_{c-m+j}^T\cdot%20v_c)-\sum_{k=1}^{K}log\sigma(-\widetilde{u_k}^T\cdot%20v_c))    

对 CBOW 模型，我们对给定上下文向量![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m})来观察中心词![](https://latex.codecogs.com/png.image?\dpi{110}%20u_c)的新的目标函数为

![](https://latex.codecogs.com/png.image?\dpi{110}%20-log\sigma(u_{c}^T\cdot%20\hat{v})-\sum_{k=1}^{K}log\sigma(-\widetilde{u_k}^T\cdot%20\hat{v}))   

在上面的公式中，![]()s是从![]()中抽样。有很多关于如何得到最好近似的讨论，从实际效果看来最好的是指数为 ¾ 的 Unigram 模型。那么为什么是 ¾？下面有一些例如可能让你有一些直观的了解：

![](https://latex.codecogs.com/png.image?\dpi{110}%20is:0.9^{3/4}=0.92)       
![](https://latex.codecogs.com/png.image?\dpi{110}%20Constitution:0.09^{3/4}=0.16)     
![](https://latex.codecogs.com/png.image?\dpi{110}%20bombastic:0.01^{3/4}=0.032)

“Bombastic”现在被抽样的概率是之前的三倍，而“is”只比之前的才提高了一点点。

### Hierarchical Softmax

Mikolov 在论文《Distributed Representations of Words and Phrases and their Compositionality.》中提出了 hierarchical softmax，相比普通的 softmax 这是一种更有效的替代方法。在实际中，**hierarchical softmax 对低频词往往表现得更好，负采样对高频词和较低维度向量表现得更好**。

Hierarchical softmax 使用一个二叉树来表示词表中的所有词。树中的每个叶结点都是一个单词，而且只有一条路径从根结点到叶结点。在这个模型中，没有词的输出表示。相反，图的每个节点（根节点和叶结点除外）与模型要学习的向量相关联。单词作为输出单词的概率定义为从根随机游走到单词所对应的叶的概率。计算成本由![](https://latex.codecogs.com/png.image?\dpi{110}%20O(|V|))变为![](https://latex.codecogs.com/png.image?\dpi{110}%20O(log(|V|)))。   

在这个模型中，给定一个向量![](https://latex.codecogs.com/png.image?\dpi{110}%20w_i)的下一个单词![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的概率为![](https://latex.codecogs.com/png.image?\dpi{110}%20P(w|w_i))，等于从根节点开始到对应![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的叶节点结束的随机漫步概率。这个方法最大的优势是计算概率的时间复杂度仅仅是![](https://latex.codecogs.com/png.image?\dpi{110}%20O(log(|V|)))，对应着路的长度。

下图是Hierarchical softmax的二叉树示意图

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-28%20at%205.21.50%20PM.png)  

让我们引入一些概念。令![](https://latex.codecogs.com/png.image?\dpi{110}%20L(w))为从根节点到叶子节点![](https://latex.codecogs.com/png.image?\dpi{110}%20w≈)的路径中节点数目。例如，上图中![](https://latex.codecogs.com/png.image?\dpi{110}%20L(w_2))为3。我们定义![](https://latex.codecogs.com/png.image?\dpi{110}%20n(w,i))为与向量![](https://latex.codecogs.com/png.image?\dpi{110}%20v_{n(w,i)})相关的路径上第![](https://latex.codecogs.com/png.image?\dpi{110}%20i)个节点。因此![](https://latex.codecogs.com/png.image?\dpi{110}%20n(w,1))是根节点，而![](https://latex.codecogs.com/png.image?\dpi{110}%20n(w,L(w)))是![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的父节点。
现在对每个内部节点![](https://latex.codecogs.com/png.image?\dpi{110}%20n)，我们任意选取一个它的子节点，定义为![](https://latex.codecogs.com/png.image?\dpi{110}%20ch(n))（一般为左节点）。然后我们计算概率为

![](https://latex.codecogs.com/png.image?\dpi{110}%20p(w|w_i)=\prod_{j=1}^{L(w)-1}\sigma%20([n(w,j+1)=ch(n(w,j))]\cdot%20v_{n(w,j)}^Tv_{w_i}))

其中

![](https://latex.codecogs.com/png.image?\dpi{110}%20[x]=\left\{\begin{matrix}1%20\%20\%20\%20\%20if%20\%20x%20\%20%20is\%20%20true\\-1%20\%20\%20\%20\%20otherwise\end{matrix}\right.)

这个公式看起来非常复杂，让我们细细梳理一下。

首先，我们将根据从根节点![](https://latex.codecogs.com/png.image?\dpi{110}%20(n(w,1)))到叶节点![](https://latex.codecogs.com/png.image?\dpi{110}%20w)的路径的形状来计算相乘的项。如果我们假设![](https://latex.codecogs.com/png.image?\dpi{110}%20ch(n))一直都是n的左节点，然后当路径往左时，
![](https://latex.codecogs.com/png.image?\dpi{110}%20[n(w,j+1)=ch(n(w,j))]) 的返回值为1，往右返回值为0.

此外![](https://latex.codecogs.com/png.image?\dpi{110}%20[n(w,j+1)=ch(n(w,j))]) 提供了归一化的作用。在节点n处，如果我们将去往左和右的节点概率相加，对于![](https://latex.codecogs.com/png.image?\dpi{110}%20v_n^Tv_{w_i})的任何值则可以检查

![](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma(v_n^Tv_{w_i})+\sigma(-v_n^Tv_{w_i})=1)

归一化保证了![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_{w=1}^{|V|}P(w|w_i)=1)，和在普通的softmax是 一样的。

最后我们计算点积来比较输入向量![](https://latex.codecogs.com/png.image?\dpi{110}%20v_{w_i})对每个内部节点向量![](https://latex.codecogs.com/png.image?\dpi{110}%20v_{n(w,j)}^T)的相似度。下面我们给出一个例子。以上图中的![](https://latex.codecogs.com/png.image?\dpi{110}%20w_2)为例，从根节点要经过两次左边和一次右边的边才能到达![](https://latex.codecogs.com/png.image?\dpi{110}%20w_2)，因此

![](https://latex.codecogs.com/png.image?\dpi{110}%20p(w|w_2)=p(n(w_2,1),left)\cdot%20p(n(w_2,2),left)%20\cdot%20p(n(w_2,3),right))    
&emsp; &emsp; &ensp;  &emsp; &ensp;  &ensp; &nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\sigma(v_{n(w_2,1)}^Tv_{w_i})\cdot%20\sigma(v_{n(w_2,1)}^Tv_{w_i})%20\cdot%20(-\sigma(v_{n(w_2,1)}^Tv_{w_i})))

我们训练模型的目标是最小化负的对数似然![](https://latex.codecogs.com/png.image?\dpi{110}%20-logP(w|w_i))。不是更新每个词的输出向量，而是更新更新二叉树中从根结点到叶结点的路径上的节点的向量。

该方法的速度由构建二叉树的方式确定，并将词分配给叶节点。Mikolov 在论文《Distributed Representations of Words and Phrases and their Compositionality.》中使用的是哈夫曼树，在树中分配高频词到较短的路径。




