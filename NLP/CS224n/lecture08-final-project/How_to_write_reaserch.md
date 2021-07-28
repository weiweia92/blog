## Practical Methodology

Successfully applying deep learning techniques requires more than just a good knowledge of what algorithms exist and the principles that explain how they work.
A good machine learning practitioner(从业者) also needs to know how to choose an algorithm for a particular application and how to monitor and respond to feedback
obtained from experiments in order to improve a machine learning system. During day-to-day development(日常开发) of machine learning systems, practitioners need to 
decide whether to gather more data,increase or decrease model capacity, add or remove regularizing features, improve the optimization(优化) of a model, improve 
approximate inference(近似推理) in a model, or debug the software implementation(软件实现) of the model. All these operations are at the very least time consuming to 
try out, so it is important to be able to determine the right course of action rather than blindly guessing.

Most of this book is about different machine learning models, training algorithms, and objective functions. This may give the impression that the most important 
ingredient to being a machine learning expert is knowing a wide variety of machine learning techniques and being good at different kinds of math. In practice, 
one can usually do much better with a correct application of a commonplace algorithm than by sloppily(马虎) applying an obscure(晦涩的) algorithm. Correct application
of an algorithm depends on mastering(掌握) some fairly(相当) simple methodology. Many of the recommendations in this chapter are adapted from Ng (2015).

We recommend the following practical design process:

* Determine your goals—what error metric(指标) to use, and your target value for this error metric. These goals and error metrics should be driven by the problem that the
application is intended to solve.

* Establish(建立) a working end-to-end pipeline as soon as possible, including the estimation of the appropriate(适当的) performance metrics。

* Instrument the system well to determine(确定、判断) bottlenecks in performance. Diagnose which components are performing worse than expected and whether 
poor performance is due to overﬁtting, underﬁtting, or a defect(缺陷) in the data or software.

* Repeatedly make incremental changes(增量更改) such as gathering new data, adjusting hyperparameters, or changing algorithms, based on speciﬁc ﬁndings from your instrumentation.

As a running example, we will use the Street View address number transcription(转录) system (Goodfellow et al., 2014d). The purpose of this application is to add buildings to Google Maps.
Street View cars photograph the buildings and record the GPS coordinates associated with each photograph. A convolutional network recognizes the address number in each photograph, 
allowing the Google Maps database to add that address in the correct location. The story of how this commercial application was developed gives an example of how to 
follow the design methodology we advocate.

We now describe each of the steps in this process.

### 11.1 Performance Metrics

Determining your goals, in terms of which error metric to use, is a necessary ﬁrst step because your error metric will guide all your future actions. 
You should also have an idea of what level of performance you desire. 

Keep in mind that for most applications, it is impossible to achieve absolute zero error. The Bayes error deﬁnes the minimum error rate that you can hope toachieve,
even if you have inﬁnite training data and can recover the true probability distribution. This is because your input features may not contain complete information about the 
output variable, or because the system might be intrinsically stochastic. You will also be limited by having a ﬁnite amount of training data.

The amount of training data can be limited for a variety of reasons. When your goal is to build the best possible real-world product or service,you can typically 
collect more data but must determine the value of reducing error further and weigh(权衡) this against the cost(成本) of collecting more data. Data collection can require time,money, 
or human suffering (for example, if your data collection process involves performing invasive(侵犯的，侵入性) medical tests). When your goal is to answer a scientiﬁc question 
about which algorithm performs better on a ﬁxed benchmark(基准), the benchmark speciﬁcation usually determines the training set, and you are not allowed to collectmore data.

How can one determine a reasonable level of performance to expect? Typically,in the academic setting(学术环境), we have some estimate of the error rate that is attainable
based on previously published benchmark results. In the real-word setting, we have some idea of the error rate that is necessary for an application to be safe,cost-effective, 
or appealing to consumers(吸引消费者). Once you have determined your realistic desired error rate, your design decisions will be guided by reaching this error rate.

Another important consideration besides the target value of the performance metric is the choice of which metric to use. Several different performance metrics may be used 
to measure the effectiveness of a complete application that includes machine learning components. These performance metrics are usually different from the cost function 
used to train the model. As described in section 5.1.2, it is common to measure the accuracy, or equivalently, the error rate, of a system.

However, many applications require more advanced metrics.

Sometimes it is much more costly(代价) to make one kind of a mistake than another. For example, an e-mail spam detection system can make two kinds of mistakes:
incorrectly classifying a legitimate(合法性) message as spam, and incorrectly allowing a spam message to appear in the inbox. It is much worse to block(阻止) a 
legitimate message than to allow a questionable message to pass through. Rather than measuring the error rate of a spam classiﬁer, we may wish to measure 
some form of total cost, where the cost of blocking legitimate messages is higher than the costof allowing spam messages.

Sometimes we wish to train a binary classiﬁer that is intended to detect somerare event. For example, we might design a medical test for a rare disease.
Suppose that only one in every million people has this disease. We can easily achieve 99.9999 percent accuracy on the detection task, by simply hard coding 
the classifier to always report that the disease is absent(缺席的). Clearly, accuracy is a poor way to characterize the performance of such a system.
One way to solve this problem is to instead measure **precision** and **recall**. Precision is the fraction(分数) of detections reported by the model that 
were correct, while recall is the fraction of true events that were detected(检测). A detector that says no one has the disease would achieveperfect precision, 
but zero recall. A detector that says everyone has the disease would achieve perfect recall, but precision equal to the percentage of people whohave the disease
(0.0001 percent in our example of a disease that only one people in a million have).  When using precision and recall, it is common to plot a PR curve,
with precision on the y-axis and recall on the x-axis. The classifier generates a score that is higher if the event to be detected occurred. For example, 
a feedforward network designed to detect a disease outputs ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}%20=%20P(y=1|x)), estimating the 
probability that a person whose medical results are described by features ![](https://latex.codecogs.com/png.image?\dpi{110}%20x) has the disease. 
We choose to report a detection whenever this score exceeds some threshold. By varying the threshold, we can trade precision for recall. In many 
cases, we wish to summarize the performance of the classifier with a single number rather than a curve. To do so, we can convert precision 
![](https://latex.codecogs.com/png.image?\dpi{110}%20p) and recall ![](https://latex.codecogs.com/png.image?\dpi{110}%20r) into an F-score given by 

![](https://latex.codecogs.com/png.image?\dpi{110}%20F%20=%20\frac{2pr}{p+r})

Another option is to report the total area(面积，区域) lying beneath(下方) the PR curve.

In some applications, it is possible for the machine learning system to refuse to make a decision. This is useful when the machine learning algorithm 
can estimate how conﬁdent it should be about a decision, especially if a wrong decision can be harmful and if a human operator is able to occasionally
take over(接管). The Street View transcription system provides an example of this situation. The task is to transcribe(转录) the address number from a 
photograph to associate the location where the photo was taken with the correct address in a map. Because the value of the map degrades considerably(大大下降)
if the map is inaccurate, it is important to add an address only if the transcription is correct. If the machine learning system thinksthat it is 
less likely than a human being to obtain the correct transcription,  then the best course of action is to allow a human to transcribe the photo instead.
Of course,the machine learning system is only useful if it is able to dramatically reduce the amount of photos that the human operators must process. 
A natural performance metric to use in this situation is  **coverage**. Coverage is the fraction of examples for which the machine learning system 
is able to produce a response. It is possible to trade coverage for accuracy. One can always obtain 100 percent accuracy by refusing to process any example, 
but this reduces the coverage to 0 percent. For the Street View task, the goal for the project was to reach human-level transcription accuracy while 
maintaining 95 percent coverage. Human-level performance on thistask is 98 percent accuracy.

Many other metrics are possible. We can, for example, measure click-through rates, collect user satisfaction surveys, and so on. Many specialized application 
areas have application-speciﬁc criteria(特定于应用的标准) as well.

What is important is to determine which performance metric to improve ahead of time, then concentrate(集中) on improving this metric. 
Without clearly defined goals,it can be difficult to tell whether changes to a machine learning system make progress or not.

### 11.2 Default Baseline Models

After choosing performance metrics and goals, the next step in any practical application is to establish a reasonable end-to-end system as soon as possible.
In this section, we provide recommendations for which algorithms to use as the ﬁrst baseline approach in various situations.  Keep in mind that deep learning
research progresses quickly, so better default algorithms are likely to become available soon after this writing. 

Depending on the complexity of your problem, you may even want to begin without using deep learning. If your problem has a chance of being solved by just choosing
a few linear weights correctly, you may want to begin with a simple statistical model like logistic regression. 

If you know that your problem falls into an “AI-complete” category like object recognition, speech recognition, machine translation, and so on, then you are likely 
to do well by beginning with an appropriate deep learning model.

First, choose the general category of model based on the structure of your data. If you want to perform supervised learning with ﬁxed-size vectors as input, use a
feedforward network with fully connected layers. If the input has known topological structure (for example, if the input is an image), use a convolutional network.
In these cases, you should begin by using some kind of piecewise(分段) linear unit (ReLUs or their generalizations, such as Leaky ReLUs, PreLus, or maxout). 
If your input or output is a sequence, use a gated recurrent net (LSTM or GRU). 

A reasonable choice of optimization algorithm is SGD with momentum with a decaying learning rate(popular decay schemes that perform better or worse on 
different problems include decaying linearly until reaching a ﬁxed minimum learning rate, decaying exponentially, or decreasing the learning rate by a factor of
2–10(学习率降低2-10倍) each time validation error plateaus(错误率平稳时)). Another reasonable alternative is Adam. Batch normalization can have a dramatic effect
on optimization performance, especially for convolutional networks and networks with sigmoidal nonlinearities. While it is reasonable to omit(省略) batch normalization
from the very ﬁrst baseline, itshould be introduced quickly if optimization appears to be problematic. 

Unless your training set contains tens of millions of examples or more, you should include some mild(轻度) forms of regularization from the start.
on optimization performance, especially for convolutional networks and networks with sigmoidal nonlinearities. While it is reasonable to omit(省略)
batch normalization from the very ﬁrst baseline, it should be introduced quickly if optimization appears to be problematic. 

Unless your training set contains tens of millions of examples or more, you should include some mild(温和) forms of regularization from the start. Early stopping 
should be used almost universally. Dropout is an excellent regularizer that is easy to implement and compatible(兼容) with many models and training algorithms. 
Batch normalization also sometimes reduces generalization error and allows dropout tobe omitted, because of the noise in the estimate of the statistics used to 
normalizeeach variable(用于归一化每个变量的统计估计中存在噪声).  

If your task is similar to another task that has been studied extensively, you will probably do well by ﬁrst copying the model and algorithm that is already 
known to perform best on the previously studied task. You may even want to copy a trained model from that task. For example, it is common to use the features 
from a convolutional network trained on ImageNet to solve other computer visiontasks (Girshick et al., 2015).   

A common question is whether to begin by using unsupervised learning, described further in part III. This is somewhat domain speciﬁc. Some domains, such as natural
language processing, are known to beneﬁt tremendously from unsupervised learning techniques, such as learning unsupervised word embeddings. In other domains, 
such as computer vision, current unsupervised learning techniques do not bring a beneﬁt, except in the semi-supervised setting, when the number of labeled examples 
is very small (Kingma et al., 2014; Rasmus et al., 2015).  If your application is in a context(在...环境下) where unsupervised learning is known to be important,
then include it in your ﬁrst end-to-end baseline. Otherwise, only use unsupervised learning in your ﬁrst attempt if the task you want to solve is unsupervised.
You can always try adding unsupervised learning later if you observe that your initialbaseline overﬁts.

### 11.3 Determining Whether to Gather More Data

After the ﬁrst end-to-end system is established, it is time to measure the performance of the algorithm and determine how to improve it. Many machine learning 
novices(新手) are tempted to make improvements by trying out many different algorithms. Yet, it is often much better to gather more data than to improve the 
learning algorithm. 

How does one decide whether to gather more data? First, determine whether the performance on the training set is acceptable. If performance on the training set 
is poor, the learning algorithm is not using the training data that is already available, so there is no reason(没有理由) to gather more data. Instead, try increasing the 
size of the model by adding more layers or adding more hidden units to each layer. Also, try improving the learning algorithm, for example by tuning the learning rate 
hyperparameter. If large models and carefully tuned optimization algorithmsdo not work well, then the problem might be the *quality* of the training data.
The data may be too noisy or may not include the right inputs needed to predict thedesired outputs. This suggests starting over(重新开始), collecting cleaner data, 
or collecting a richer set of features.  

If the performance on the training set is acceptable, then measure the performance on a test set. If the performance on the test set is also acceptable,then there 
is nothing left to be done. If test set performance is much worse than training set performance, then gathering more data is one of the most effective solutions. 
The key considerations are the cost and feasibility of gathering more data, the cost and feasibility of reducing the test error by other means, and the amount of 
data that is expected(预测) to be necessary to improve test set performance signiﬁcantly. At large internet companies with millions or billions of users, it is 
feasible to gather large datasets, and the expense of doing so can be considerably less than that of the alternatives, so the answer is almost always to gather more
training data. For example, the development of large labeled datasets was one of the most important factors in solving object recognition. In other contexts, such as ‘
medical applications, it may be costly or infeasible to gather more data. A simple alternative to gathering more data is to reduce the size of the model or improve 
regularization, by adjusting hyperparameters such as weight decay coeﬃcients, or by adding regularization strategies such as dropout. If you ﬁnd that the gap between 
train and test performance is still unacceptable even after tuning the regularization hyperparameters, then gathering more data is advisable.  

When deciding whether to gather more data, it is also necessary to decide how much to gather. It is helpful to plot curves showing the relationship between training set 
size and generalization error, as in ﬁgure 5.4. By extrapolating(外推) such curves, one can predict how much additional training data would be needed to achieve a 
certain level of performance. Usually, adding a small fraction of the total number of examples will not have a noticeable(显著的) effect on generalization error.
It is therefore recommended to experiment with training set sizes on a logarithmic scale, for example, doubling the number of examples between consecutive experiments
(连续实验). 

If gathering much more data is not feasible, the only other way to improve generalization error is to improve the learning algorithm itself. This becomes the domain 
of research and not the domain of advice for applied practitioners. 

### 11.4 Selecting Hyperparameters

Most deep learning algorithms come with several hyperparameters that control many aspects of the algorithm’s behavior. Some of these hyperparameters affect the time 
and memory cost of running the algorithm. Some of these hyperparameters affect the quality of the model recovered by the training process and its ability to infer(推断) 
correct results when deployed(部署) on new inputs.  

There are two basic approaches to choosing these hyperparameters: choosing them manually and choosing them automatically. Choosing the hyperparameters manually 
requires understanding what the hyperparameters do and how machine learning models achieve good generalization. Automatic hyperparameter selection algorithms 
greatly reduce the need to understand these ideas, but they are oftenmuch more computationally costly.  

### 11.4.1 Manual Hyperparameter Tuning

To set hyperparameters manually, one must understand the relationship between hyperparameters, training error, generalization error and computational resources
(memory and runtime). This means establishing a solid foundation on the fundamental ideas concerning the effective capacity of a learning algorithm, 
as described in chapter 5.  

The goal of manual hyperparameter search is usually to ﬁnd the lowest generalization error subject to(受制于) some runtime and memory budget(预算).We do not 
discuss how to determine the runtime and memory impact of various hyperparameters here because this is highly platform dependent(依赖).

The primary goal of manual hyperparameter search is to adjust the effective capacity of the model to match the complexity of the task. Effective capacityis 
constrained by three factors: the representational capacity of the model, the ability of the learning algorithm to successfully minimize the cost function 
used to train the model, and the degree to which the cost function and training procedure(程序) regularize the model. A model with more layers and more 
hidden units per layer has higher representational capacity—it is capable of representing more complicated functions. It cannot necessarily learn all 
these functions though, if the training algorithm cannot discover that certain functions do a good job of minimizing the training cost, or if regularization 
terms such as weight decay forbid(禁止) some of these functions. 

The generalization error typically follows a U-shaped curve when plotted as a function of one of the hyperparameters, as in ﬁgure 5.3. At one extreme(极端), 
the hyperparameter value corresponds to low capacity, and generalization error is high because training error is high. This is the underﬁtting regime.
At the other extreme,the hyperparameter value corresponds to high capacity, and the generalization error is high because the gap between training and test error 
is high. Somewhere in the middle lies the optimal model capacity, which achieves the lowest possible generalization error, by adding a medium generalization gap 
to a medium amount of training error. 

For some hyperparameters, overﬁtting occurs when the value of the hyperparameter is large. The number of hidden units in a layer is one such example, because 
increasing the number of hidden units increases the capacity of the model. For some hyperparameters, overﬁtting occurs when the value of the hyperparameter 
is small. For example, the smallest allowable weight decay coeﬃcient of zero corresponds to the greatest effective capacity of the learning algorithm. 

Not every hyperparameter will be able to explore the entire U-shaped curve. Many hyperparameters are discrete, such as the number of units in a layer or the 
number of linear pieces in a maxout unit, so it is only possible to visit a few points along the curve. Some hyperparameters are binary. Usually these 
hyperparameters are switches that specify whether or not to use some optional component of the learning algorithm, such as a preprocessing step that normalizes the 
input features by subtracting their mean and dividing by their standard deviation. These hyperparameters can explore only two points on the curve. 
Other hyperparameters have some minimum or maximum value that prevents them from exploring some part of the curve. For example, the minimum weight decay coeﬃcient 
is zero. This means that if the model is underﬁtting when weight decay is zero, we cannot enter the overﬁtting region by modifying the weight decay coeﬃcient.
In other words,some hyperparameters can only subtract capacity.  

The learning rate is perhaps the most important hyperparameter. If you have time to tune only one hyperparameter, tune the learning rate. It controls the effective 
capacity of the model in a more complicated way than other hyperparameters—the effective capacity of the model is highest when the learning rate is *correct* for 
the optimization problem, not when the learning rate is especially large or especially small. The learning rate has a U-shaped curve for training error,illustrated 
in ﬁgure
![]()















