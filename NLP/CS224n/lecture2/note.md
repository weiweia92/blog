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
![](https://latex.codecogs.com/png.image?\dpi{110}%20J_t(\theta)=log\sigma(u_o^Tv_c)+\sum_{i=1}^{k}E_{j\sim%20P(w)}[log\sigma(-u_j^Tv_c)])   

本课以及作业中的目标函数是

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{neg-sample}(o,v_c,U)=-log(\sigma(u_o^Tv_c))-\sum_{k=1}^Klog(\sigma(-u_k^Tv_c)))

* 我们希望中心词与真实上下文单词的向量点积更大，与随机单词的点积更小。
* k是我们负采样的样本数目

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(w)=U(w)^{3/4}/Z)

使用上式为抽样分布，![]()是unigram分布，通过3/4次方，相对减少常见单词的频率，增大稀有词的概率。Z用于生成概率分布。

### But why not capture co-occurrence counts directly?

共现矩阵 X

* 两个选项：windows vs. full document   

* Window ：与word2vec类似，在每个单词周围都使用Window，包括语法(POS)和语义信息

* Word-document 共现矩阵的基本假设是在同一篇文章中出现的单词更有可能相互关联。假设单词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20i)出现在文章 ![](https://latex.codecogs.com/png.image?\dpi{110}%20j)中，则矩阵元素 ![](https://latex.codecogs.com/png.image?\dpi{110}%20X_{ij}) 加一，当我们处理完数据库中的所有文章后，就得到了矩阵![](https://latex.codecogs.com/png.image?\dpi{110}%20X) ，其大小为 ![](https://latex.codecogs.com/png.image?\dpi{110}%20|V|\times%20M)，其中 ![](https://latex.codecogs.com/png.image?\dpi{110}%20|V|)为词汇量，而 ![](https://latex.codecogs.com/png.image?\dpi{110}%20M)为文章数。这一构建单词文章co-occurrence matrix的方法也是经典的Latent Semantic Analysis(潜在语义分析)所采用的。

利用某个定长窗口中单词与单词同时出现的次数来产生window-based (word-word) co-occurrence matrix。下面以窗口长度为1来举例，假设我们的数据包含以下几个句子：

* I like deep learning.
* I like NLP.
* I enjoy flying.

则我们可以得到如下的word-word co-occurrence matrix:    

![]()   

使用共现次数衡量单词的相似性，但是会随着词汇量的增加而增大矩阵的大小，并且需要很多空间来存储这一高维矩阵，后续的分类模型也会由于矩阵的稀疏性而存在稀疏性问题，使得效果不佳。我们需要对这一矩阵进行降维，获得低维(25-1000)的稠密向量。

**Method 1: Dimensionality Reduction on X (HW1)**

使用SVD方法将共现矩阵 X 分解为![](https://latex.codecogs.com/png.image?\dpi{110}%20U%20\Sigma%20V^T),![](https://latex.codecogs.com/png.image?\dpi{110}%20\Sigma)是对角线矩阵，对角线上的值是矩阵的奇异值。![](https://latex.codecogs.com/png.image?\dpi{110}%20U), ![](https://latex.codecogs.com/png.image?\dpi{110}%20V)是对应于行和列的正交基。

为了减少尺度同时尽量保存有效信息，可保留对角矩阵的最大的k个值，并将矩阵![](https://latex.codecogs.com/png.image?\dpi{110}%20U),![](https://latex.codecogs.com/png.image?\dpi{110}%20V)的相应的行列保留。这是经典的线性代数算法，对于大型矩阵而言，计算代价昂贵。  

**Method 2: Hacks to X (several used in Rohde et al.2005)**

按比例调整 counts 会很有效

* 对高频词进行缩放(语法有太多的影响)

   * 使用log进行缩放

   * ![](https://latex.codecogs.com/png.image?\dpi{110}%20min(X,t),t\approx%20100)

   * 直接全部忽视

* 在基于window的计数中，提高更加接近的单词的计数
* 使用Person相关系数

Conclusion：对计数进行处理是可以得到有效的词向量的

基于计数：使用整个矩阵的全局统计数据来直接估计

* 优点
   * 训练快速
   * 统计数据高效利用
* 缺点
   * 主要用于捕捉单词相似性
   * 对大量数据给予比例失调的重视

转换计数：定义概率分布并试图预测单词

* 优点
   * 提高其他任务的性能
   * 能捕获除了单词相似性以外的复杂的模式
* 缺点
   * 与语料库大小有关的量表
   * 统计数据的低效使用（采样是对统计数据的低效使用）

### Encoding meaning in vector differences

关键思想：共现概率的比值可以对meaning component进行编码。重点不是单一的概率大小，重点是他们之间的比值，其中蕴含着meaning component。

![]()

## Notes 02 GloVe, Evaluation and Training

### 1 Global Vectors for Word Representation (GloVe)

**1.1 Comparison with Previous Methods**

到目前为止，我们已经研究了两类主要的词嵌入方法。第一类是基于统计并且依赖矩阵分解（例如LSA，HAL）。虽然这类方法有效地利用了**全局的信息**，它们主要用于捕获单词的**相似性**，但是对例如单词类比的任务上表现不好。另外一类方法是基于浅层窗口（例如，Skip-Gram 和 CBOW 模型），这类模型通过在局部上下文窗口通过预测来学习词向量。这些模型除了在单词相似性任务上表现良好外，还展示了捕获**复杂语言模式**能力，但未能利用到全局共现统计数据。

相比之下，GloVe 由一个加权最小二乘模型组成，基于全局word-word共现计数进行训练，从而有效地利用全局统计数据。模型生成了包含有意义的子结构的单词向量空间，在词类比任务上表现非常出色。

Glove 利用全局统计量，以最小二乘为目标，预测单词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20j)出现在单词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20i)上下文中的概率

**1.2 Co-occurrence Matrix**

![](https://latex.codecogs.com/png.image?\dpi{110}%20X)表示 word-word 共现矩阵，其中![](https://latex.codecogs.com/png.image?\dpi{110}%20X_{ij})表示词![](https://latex.codecogs.com/png.image?\dpi{110}%20j)出现在词![](https://latex.codecogs.com/png.image?\dpi{110}%20i)的上下文的次数。令 ![](https://latex.codecogs.com/png.image?\dpi{110}%20X_i%20=%20\sum_{k}X_{ik})为任意词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20k)出现在词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20i)的上下文的次数。最后，令 ![](https://latex.codecogs.com/png.image?\dpi{110}%20P_{ij}=P(w_j|w_i)=\frac{X_{ij}}{X_i})是词![](https://latex.codecogs.com/png.image?\dpi{110}%20j)出现在词![](https://latex.codecogs.com/png.image?\dpi{110}%20i)的上下文的概率。

计算这个矩阵需要遍历一次整个语料库获得统计信息。对庞大的语料库，这样的遍历会产生非常大的计算量，但是这只是一次性的前期投入成本。

**1.3 Least Squares Objective**

回想一下 Skip-Gram 模型，我们使用 softmax 来计算词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20j)出现在词 ![](https://latex.codecogs.com/png.image?\dpi{110}%20i)的上下文的概率

![](https://latex.codecogs.com/png.image?\dpi{110}%20Q_{ij}=\frac{exp(u_j^Tv_i)}{\sum_{w=1}^Wexp(u_w^Tv_i)})

训练时以在线随机的方式进行，但是隐含的全局交叉熵损失可以如下计算：

![](https://latex.codecogs.com/png.image?\dpi{110}%20J%20=%20-\sum_{i\in%20corpus}\sum_{j%20\in%20context(i)}logQ_{ij})

同样的单词![](https://latex.codecogs.com/png.image?\dpi{110}%20i)和![](https://latex.codecogs.com/png.image?\dpi{110}%20j)可能在语料库中出现多次，因此首先将![](https://latex.codecogs.com/png.image?\dpi{110}%20i)和![](https://latex.codecogs.com/png.image?\dpi{110}%20j)相同的值组合起来更有效：

![](https://latex.codecogs.com/png.image?\dpi{110}%20J%20=%20-\sum_{i=1}^{W}\sum_{j=1}^{W}X_{ij}logQ_{ij}) 

其中，共现频率的值是通过共现矩阵![](https://latex.codecogs.com/png.image?\dpi{110}%20X)给定。

交叉熵损失的一个显着缺点是要求分布 ![](https://latex.codecogs.com/png.image?\dpi{110}%20Q)被正确归一化，因为对整个词汇的求和的计算量是非常大的。因此，我们使用一个最小二乘的目标函数，其中 ![](https://latex.codecogs.com/png.image?\dpi{110}%20P)和 ![](https://latex.codecogs.com/png.image?\dpi{110}%20Q)的归一化因子被丢弃了：

![](https://latex.codecogs.com/png.image?\dpi{110}%20\widehat{J}=\sum_{i=1}^W\sum_{j=1}^WX_i(\widehat{P_{ij}}-\widehat{Q_{ij}})^2)

其中![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{P_{ij}}=X_{ij})和![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{Q_{ij}}=exp(\vec{u_j}^T\vec{v_i}))是未对一化分布。这个公式带来了一个新的问题，![](https://latex.codecogs.com/png.image?\dpi{110}%20X_{ij})经常会是很大的值，从而难以优化。一个有效的改变是最小化![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{P})和![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{Q})对数的平方误差：

![](https://latex.codecogs.com/png.image?\dpi{110}%20\widehat{J}=\sum_{i=1}^W\sum_{j=1}^WX_i(log(\hat{p_{ij}})-log(\hat{Q_{ij}}))^2)

&emsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\sum_{i=1}^W\sum_{j=1}^WX_i(u_j^Tv_i-logX_{ij})^2)

另外一个问题是权值因子![](https://latex.codecogs.com/png.image?\dpi{110}%20X_i)不能保证是最优的。因此，我们引入更一般化的权值函数，我们可以自由地依赖于上下文单词：

![](https://latex.codecogs.com/png.image?\dpi{110}%20\widehat{J}=\sum_{i=1}^{W}\sum_{j=1}^{W}f(X_{ij})(u_j^Tv_i-logX_{ij})^2)

**1.4 Conclusion**

总而言之，GloVe 模型仅对单词共现矩阵中的非零元素训练，从而有效地利用全局统计信息，并生成具有有意义的子结构向量空间。给出相同的语料库，词汇，窗口大小和训练时间，它的表现都优于 word2vec，它可以更快地实现更好的效果，并且无论速度如何，都能获得最佳效果。

### 2 Evaluation of Word Vectors

到目前为止，我们已经讨论了诸如 Word2Vec 和 GloVe 来训练和发现语义空间中的自然语言词语的潜在向量表示。在这部分，我们讨论如何量化评估词向量的质量。

**2.1 Intrinsic Evaluation** 

词向量的内部评估是对一组由如 Word2Vec 或 GloVe 生成的词向量在特定的中间子任务（如词类比）上的评估。这些子任务通常简单而且计算速度快，从而能够帮助我们理解生成的的词向量。内部评估通常应该返回给我们一个数值，来表示这些词向量在评估子任务上的表现。

* 对特定的中间任务进行评估
* 可以很快的计算性能
* 帮助理解子系统
* 需要和真实的任务正相关来确定有用性

在下图中，左子系统（红色）训练的计算量大，因此更改为一个简单的子系统（绿色）作内部评估。

![]()

**动机**：我们考虑创建一个问答系统，其中使用词向量作为输入的例子。一个方法是训练一个机器学习系统：

1. 输入词语    
2. 将输入词语转换为词向量     
3. 对一个复杂的机器学习系统，使用词向量作为输入     
4. 将输出的词向量通过系统映射到自然语言词语上     
5. 生成词语作为答案
     
当然，在训练这样的一个问答系统的过程中，因为它们被用在下游子系统（例如深度神经网络），我们需要创建最优的词向量表示。在实际操作中，我们需要对 Word2Vec 子系统中的许多超参数进行调整（例如词向量的维度）。

虽然最理想的方法是在 Word2Vec 子系统中的任何参数改变后都重新训练，但从工程角度来看是不实际的，因为机器学习系统（在第3步）通常是一个深层神经网络，网络中的数百万个参数需要很长的时间训练。

在这样的情况下，我们希望能有一个简单的内部评估技术来度量词向量子系统的好坏。显然的要求是内部评价与最终任务的表现有正相关关系。

**2.2 Extrinsic Evaluation**

词向量的外部评估是对一组在实际任务中生成的词向量的评估。这些任务通常复杂而且计算速度慢。对我们上面的例子，允许对问题答案进行评估的系统是外部评估系统。通常，优化表现不佳的外部评估系统我们难以确定哪个特定子系统存在错误，这就需要进一步的内部评估。

* 对真实任务的评估     
* 计算性能可能很慢      
* 不清楚是子系统出了问题，还是其他子系统出了问题，还是内部交互出了问题    
* 如果替换子系统提高了性能，那么更改可能是好的
      
**2.3 Intrinsic Evaluation Example: Word Vector Analogies**

一个比较常用的内部评估的方法是词向量的类比。在词向量类比中，给定以下形式的不完整类比：

![](https://latex.codecogs.com/png.image?\dpi{110}%20a:b::c:?)

然后内部评估系统计算词向量的最大余弦相似度：

![](https://latex.codecogs.com/png.image?\dpi{110}%20d=\underset{i}{argmax}\frac{(x_b-x_a+x_c)^Tx_i}{|x_b-x_a+x_c|})

**2.4 Intrinsic Evaluation Tuning Example: Analogy Evaluations**

我们现在探讨使用内在评估系统（如类比系统）来调整的词向量嵌入技术（如 Word2Vec 和 GloVe）中的超参数。

* 模型的表现高度依赖模型所使用的词向量的模型
   * 这点是可以预料到的，因为不同的生成词向量方法是基于不同的**特性**的（例如共现计数，奇异向量等等）。
* 语料库更大模型的表现更好
   * 这是因为模型训练的语料越大，模型的表现就会更好。例如，如果训练的时候没有包含测试的词语，那么词类比会产生错误的结果。
* 对于极高或者极低维度的词向量，模型的表现较差
   * 低维度的词向量不能捕获在语料库中不同词语的意义。这可以被看作是我们的模型复杂度太低的高偏差问题。例如，我们考虑单词“king”、“queen”、“man”、“woman”。直观上，我们需要使用例如“性别”和“领导”两个维度来将它们编码成 2 字节的词向量。维度较低的词向量不会捕获四个单词之间的语义差异，而过高的维度的可能捕获语料库中无助于泛化的噪声-即所谓的高方差问题。

**2.5 Intrinsic Evaluation Example: Correlation Evaluation**

另外一个评估词向量质量的简单方法是，让人去给两个词的相似度在一个固定的范围内（例如 0-10）评分，然后将其与对应词向量的余弦相似度进行对比。这已经在包含人为评估的各种数据集上尝试过。

**2.6 Further Reading: Dealing With Ambiguity**

我们想知道如何处理在不同的自然语言处理使用场景下，用不同的的词向量来捕获同一个单词在不同场景下的不同用法。例如，“run”是一个名词也是一个动词，在不同的语境中它的词性也会不同。论文 Improving Word Representations Via Global Context And Multiple Word Prototypes 提出上述问题的的解决方法。该方法的本质如下：

1. 收集所有出现的单词的固定大小的上下文窗口(例如前 5 个和后 5 个)。
2. 每个上下文使用上下文词向量的加权平均值来表示(使用idf加权)。
3. 用 spherical k-means 对这些上下文表示进行聚类。
4. 最后，每个单词的出现都重新标签为其相关联的类，同时对这个类，来训练对应的词向量。

要对这个问题进行更严谨的处理，可以参考原文。

### 3 Training for Extrinsic Tasks

到目前为止，我们一直都关注于内在任务，并强调其在开发良好的词向量技术中的重要性。但是大多数实际问题的最终目标是将词向量结果用于其他的外部任务。接下来会讨论处理外部任务的方法。

**3.1 Problem Formulation**

很多 NLP 的外部任务都可以表述为分类任务。例如，给定一个句子，我们可以对这个句子做情感分类，判断其情感类别为正面，负面还是中性。相似地，在命名实体识别(NER)，给定一个上下文和一个中心词，我们想将中心词分类为许多类别之一。对输入，“Jim bought 300 shares of Acme Corp. in 2006”，我们希望有这样的一个分类结果：

![](https://latex.codecogs.com/png.image?\dpi{110}%20[Jim]_{person}\%20%20bought\%20%20300\%20%20shares\%20of\%20%20[Acme%20Corp.]_{Organization}\%20in%20[2006]_{Time}.)

对这类问题，我们一般有以下形式的训练集：

![](https://latex.codecogs.com/png.image?\dpi{110}%20\{%20x^{(i)},y^{(i)}\}_{1}^{N})

其中![](https://latex.codecogs.com/png.image?\dpi{110}%20x^{(i)})是一个d维的词向量，![](https://latex.codecogs.com/png.image?\dpi{110}%20y^{(i)})是一个C维的one-hot向量，表示我们希望最终预测的标签（情感，其他词，专有名词，买／卖决策等）。我们可以使用诸如逻辑回归和 SVM 之类的算法对 2-D 词向量来进行分类

在一般的机器学习任务中，我们通常固定输入数据和目标标签，然后使用优化算法来训练权重（例如梯度下降，L-BFGS，牛顿法等等）。然而在 NLP 应用中，我们引入一个新的思想：在训练外部任务时对输入字向量进行再训练。下面我们讨论何时使用以及为什么要这样做。

**对于大型训练数据集，应考虑字向量再训练。对于小数据集，重新训练单词向量可能会降低性能。**

**3.2 Retraining Word Vectors**

正如我们迄今所讨论的那样，我们用于外部评估的词向量是通过一个简单的内部评估来进行优化并初始化。在许多情况下，这些预训练的词向量在外部评估中表现良好。但是，这些预训练的词向量在外部评估中的表现仍然有提高的可能。然而，重新训练存在着一定的风险。

如果我们在外部评估中重新训练词向量，这就需要保证训练集足够大并能覆盖词汇表中大部分的单词。这是因为Word2Vec或GloVe会生成语义相关的单词，这些单词位于单词空间的同一部分。

假设预训练向量位于二维空间中，如下图所示。在这里，我们看到在一些外部分类任务中，单词向量被正确分类。

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%202.59.57%20PM.png)

现在，如果我们因为有限的训练集大小而只对其中两个向量进行再训练，那么我们在下图中可以看到，其中一个单词被错误分类了，因为单词向量更新导致边界移动。

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%203.00.05%20PM.png)

因此，如果训练数据集很小，就不应该对单词向量进行再训练。如果培训集很大，再培训可以提高性能。

**Softmax Classification and Regularization**

我们考虑使用 Softmax 分类函数，函数形式如下所示：

![](https://latex.codecogs.com/png.image?\dpi{110}%20p(y_j=1|x)=\frac{exp(W_{j.}x)}{\sum_{c=1}^Cexp(W_{c.}x)})

这里我们计算词向量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20x)是类别![](https://latex.codecogs.com/png.image?\dpi{110}%20j)的概率。使用交叉熵损失函数计算一个样本的损失如下所示：

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{j=1}^Cy_jlog(p(y_j=1|x))=-\sum_{j=1}^Cy_jlog(\frac{exp(W_{j.}x)}{\sum_{c=1}^{C}exp(W_{c.}x)}))

当然，上述求和是对 ![](https://latex.codecogs.com/png.image?\dpi{110}%20(C-1))个零值求和，因为 ![](https://latex.codecogs.com/png.image?\dpi{110}%20y_j)仅在单个索引为 1，这意味着 ![](https://latex.codecogs.com/png.image?\dpi{110}%20x)仅属于 1 个正确的类别。现在我们定义 ![](https://latex.codecogs.com/png.image?\dpi{110}%20k)为正确类别的索引。因此，我们现在可以简化损失函数：

![](https://latex.codecogs.com/png.image?\dpi{110}%20-log(\frac{exp(W_{k.}x)}{\sum_{c=1}^Cexp(W_{c.}x)}))

然后我们可以扩展为有 ![](https://latex.codecogs.com/png.image?\dpi{110}%20N)个单词的损失函数：

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{i=1}^Nlog(\frac{exp(W_{k(i).}x^{(i)})}{\sum_{c=1}^Cexp(W_{c.}x^{(i)})}))

上面公式唯一不同是![](https://latex.codecogs.com/png.image?\dpi{110}%20k(i))是一个函数，返回![](https://latex.codecogs.com/png.image?\dpi{110}%20x(i))对应的正确类的索引

现在我们来估计一下同时训练模型的权值 ![](https://latex.codecogs.com/png.image?\dpi{110}%20W)和词向量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20x)时需要更新的参数的数量。我们知道一个简单的线性决策模型至少需要一个![](https://latex.codecogs.com/png.image?\dpi{110}%20d)维的词向量输入和生成一个 ![](https://latex.codecogs.com/png.image?\dpi{110}%20C)个类别的分布。因此更新模型的权值，我们需要 ![](https://latex.codecogs.com/png.image?\dpi{110}%20C\cdot%20d)个参数。如果我们也对词汇表 ![](https://latex.codecogs.com/png.image?\dpi{110}%20V)中的每个单词都更新词向量，那么就要更新 ![](https://latex.codecogs.com/png.image?\dpi{110}%20V)个词向量，每一个的维度是 ![](https://latex.codecogs.com/png.image?\dpi{110}%20d)维。因此对一个简单的线性分类模型，总共的参数数目是 ![](https://latex.codecogs.com/png.image?\dpi{110}%20C\cdot%20d+|V|\cdot%20d)

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%203.40.12%20PM.png)

对于一个简单的模型来说，这是相当大的参数量——这样的参数量很可能会出现过拟合的问题。

为了降低过拟合的风险，我们引入一个正则项，从贝叶斯派的思想看，这个正则项是对模型的参数加上一个先验分布，让参数变小（即接近于 0）：

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{i=1}^{N}log(\frac{exp(W_{k(i).}x^{(i)})}{\sum%20exp(W_{c.}x^{(i)})})+\lambda\sum_{k=1}^{Cd+|V|d}\theta_{k}^2)

如果调整好目标权重 ![](https://latex.codecogs.com/png.image?\dpi{110}%20\lambda)的值，最小化上面的函数将会降低出现很大的参数值的可能性，同时也提高模型的泛化能力。在我们使用更多参数更复杂的模型（例如神经网络）时，就更加需要正则化的思想。

**3.4 Window Classification**

下图是我们有一个中心词和一个长度为 2 的对称窗口。这样的上下文可以帮助分辨 Paris 是一个地点还是一个名字。

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%203.40.30%20PM.png)

目前为止，我们主要探讨了使用单个单词向量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20x)预测的外部评估任务。在现实中，因为自然语言处理的性质，这几乎不会有这样的任务。在自然语言处理中，常常存在着一词多义的情况，我们一般要利用词的上下文来判断其不同的意义。例如，如果你要某人解释“to sanction”是什么意思，你会马上意识到根据“to sanction”的上下文其意思可能是“to permit”或者“to punish”。在更多的情况下，我们使用一个单词序列作为模型的输入。这个序列是由中心词向量和上下文词向量组成。上下文中的单词数量也被称为上下文窗口大小，并根据解决的问题而变化。一般来说,较窄的窗口大小会导致在句法测试中更好的性能,而更宽的窗口会导致在语义测试中更好的性能。

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture1/a1/imgs/Screen%20Shot%202021-06-29%20at%203.43.09%20PM.png)
 
