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
![](https://latex.codecogs.com/png.image?\dpi{110}%20p) and recall ![](https://latex.codecogs.com/png.image?\dpi{110}%20r) into an F-score given by ![](https://latex.codecogs.com/png.image?\dpi{110}%20F%20=%20\frac{2pr}{p+r}).        
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
in ﬁgure.     
![](https://github.com/weiweia92/blog/blob/main/NLP/pic/Screen%20Shot%202021-07-28%20at%204.27.34%20PM.png)    

When the learning rate is too large, gradient descent can inadvertently(无意识地) increase rather than decrease the training error. In the idealized quadratic case, this occurs if the learning rate is at least twice as large as itsoptimal value (LeCun et al., 1998a). When the learning rate is too small, trainingis not only slower but may become permanently(永久地) stuck with a high training error. This effect is poorly understood (it would not happen for a convex loss function).  

Tuning the parameters other than the learning rate requires monitoring both training and test error to diagnose whether your model is overﬁtting or underﬁtting,then adjusting its capacity appropriately. 

If your error on the training set is higher than your target error rate, you have no choice but to increase capacity. If you are not using regularization and you are conﬁdent that your optimization algorithm is performing correctly, then you must add more layers to your network or add more hidden units. Unfortunately, this increases the computational costs associated with the model.

If your error on the test set is higher than your target error rate, you can now take two kinds of actions. The test error is the sum of the training error and the gap between training and test error. The optimal test error is found by trading off these quantities. Neural networks typically perform best when the training error is very low (and thus, when capacity is high) and the test error is primarily driven by the gap between training and test error. Your goal is to reduce this gap without increasing training error faster than the gap decreases. To reduce the gap,change regularization hyperparameters to reduce effective model capacity,  such asby adding dropout or weight decay. Usually the best performance comes from a large model that is regularized well, for example, by using dropout. 

Most hyperparameters can be set by reasoning about whether they increase or decrease model capacity. Some examples are included in table 11.1.    
![](https://github.com/weiweia92/blog/blob/main/NLP/pic/Screen%20Shot%202021-07-28%20at%204.35.05%20PM.png)

While manually tuning hyperparameters, do not lose sight of your end goal: good performance on the test set. Adding regularization is only one way to achieve this goal. As long as you have low training error, you can always reduce generalization error by collecting more training data. The brute force way to practically guarantee success is to continually increase model capacity and training set size until the task is solved. This approach does of course increase the computational cost of training and inference, so it is only feasible given appropriate resources. In principle, this approach could fail due to optimization difficulties, but for many problems optimization does not seem to be a significant barrier, provided that the model is chosen appropriately.

### 11.4.2 Automatic Hyperparameter Optimization Algorithms

The ideal learning algorithm just takes a dataset and outputs a function, without requiring hand tuning of hyperparameters. The popularity of several learning algorithms such as logistic regression and SVMs stems in part from their ability to perform well with only one or two tuned hyperparameters. Neural networks can sometimes perform well with only a small number of tuned hyperparameters,  but often beneﬁt significantly from tuning of forty(40次) or more. Manual hyperparameter tuning can work very well when the user has a good starting point, such as one determined by others having worked on the same type of application and architecture(由其他人在相同类型的应用程序或架构上工作过), or when the user has months or years of experience in exploring hyperparameter values for neural networks applied to similar tasks. For many applications, however, these starting points are not available. In these cases,automated algorithms can ﬁnd useful values of the hyperparameters.  

If we think about the way in which the user of a learning algorithm searches forgood values of the hyperparameters, we realize that an optimization is taking place(发生): we are trying to ﬁnd a value of the hyperparameters that optimizes an objective function, such as validation error, sometimes under constraints (such as a budget for training time, memory or recognition time). It is therefore possible, in principle,to develop hyperparameter optimization algorithms that wrap a learning algorithm and choose its hyperparameters,  thus hiding the hyperparameters of the learning algorithm from the user. Unfortunately, hyperparameter optimization algorithms often have their own hyperparameters, such as the range of values that should be explored for each of the learning algorithm’s hyperparameters. These secondary hyperparameters are usually easier to choose, however,in the sense that acceptable performance may be achieved on a wide range of tasks using the same secondary hyperparameters for all tasks.  

### 11.4.3 Grid Search

When there are three or fewer hyperparameters, the common practice is to perform **grid search**. For each hyperparameter, the user selects a small finite set of values to explore. The grid search algorithm then trains a model for every joint specification of hyperparameter values in the Cartesian product(笛卡尔积) of the set of values for each individual hyperparameter. The experiment that yields the best validationset error is then chosen as having found the best hyperparameters. See the left of figure 11.2 for an illustration of a grid of hyperparameter values.  
![](https://github.com/weiweia92/blog/blob/main/NLP/pic/Screen%20Shot%202021-07-28%20at%205.00.52%20PM.png)   

How should the lists of values to search over be chosen? In the case of numerical(ordered) hyperparameters, the smallest and largest element of each list is chosen conservatively(保守的), based on prior experience with similar experiments, to make sure that the optimal value is likely to be in the selected range.
Typically, a grid search involves picking values approximately on a *logarithmic* scale, e.g., a learning rate taken within the set ![](https://latex.codecogs.com/png.image?\dpi{110}%20\{0.1,%200.01,%2010^{-3},%2010^{-4},%2010^{-5}\}), or a number of hidden units taken with the set ![](https://latex.codecogs.com/png.image?\dpi{110}%20\{50,%20100,%20200,%20500,%201000,%202000\}).

Grid search usually performs best when it is performed(执行) repeatedly. For example,suppose that we ran a grid search over a hyperparameter ![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha) using values of
![](https://latex.codecogs.com/png.image?\dpi{110}%20\{-1,%200,%201\}). If the best value found is 1, then we under estimated the range in which the best ![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha) lies and should shift(移动) the grid and run another search with ![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha) in, for example, ![](https://latex.codecogs.com/png.image?\dpi{110}%20\{1,%202,%203\}). If we ﬁnd that the best value of ![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha) is 0, then we may wish to refine our estimate by zooming in and running a grid search over ![](https://latex.codecogs.com/png.image?\dpi{110}%20\{-0.1,%200,0.1%20\}). 

The obvious problem with grid search is that its computational cost grows exponentially with the number of hyperparameters. If there are ![](https://latex.codecogs.com/png.image?\dpi{110}%20m) hyperparameters,each taking at most ![](https://latex.codecogs.com/png.image?\dpi{110}%20n) values, then the number of training and evaluation trials required grows as ![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n^m)). The trials may be run in parallel and exploit loose(松散的) parallelism (with almost no need for communication between different machines carrying out(进行) the search). Unfortunately, because of the exponential cost of grid search, even parallelization may not provide a satisfactory size of search.

### 11.4.4 Random Search

Fortunately, there is an alternative to grid search that is as simple to program, more convenient to use, and converges much faster to good values of the hyperparameters:random search (Bergstra and Bengio, 2012). 

A random search proceeds as follows. First we define a marginal(边缘) distributionfor each hyperparameter, for example, a Bernoulli or multinoulli for binary or  discrete hyperparameters, or a uniform distribution on a log-scale for positivereal-valued hyperparameters. For example,     
`log_learning_rate ~ u(-1, -5)`   

`learning_rate = 10^{log_learning_rate}`      
where u(a, b) indicates a sample of the uniform distribution in the interval (a, b).Similarly the `log_number_of_hidden_units` may be sampled from u(log(50), log(2000)).

Unlike in a grid search, we should not discretize or bin(分箱) the values of the hyperparameters, so that we can explore a larger set of values and avoid additional computational cost. In fact, as illustrated in ﬁgure 11.2, a random search can be exponentially more efficient than a grid search, when there are several hyperparameters that do not strongly affect the performance measure. This is studied at length in Bergstra and Bengio (2012), who found that random search reduces the validation set error much faster than grid search, in terms of the number of trials(实验) run by each method.

As with grid search, we may often want to run repeated versions of random search, to reﬁne the search based on the results of the ﬁrst run.  

The main reason that random search finds good solutions faster than grid search is that it has no wasted experimental(实验) runs, unlike in the case of grid search, when two values of a hyperparameter (given values of the other hyperparameters) would give the same result. In the case of grid search, the other hyperparameters would have the same values for these two runs, whereas with random search, they would usually have different values.  Hence if the change between these two values does not marginally make much diﬀerence in terms of validation set error, grid search will unnecessarily repeat two equivalent experiments while random search will still give two independent explorations of the other hyperparameters.

### 11.4.5 Model-Based Hyperparameter Optimization

The search for good hyperparameters can be cast as(作为) an optimization problem. The decision variables are the hyperparameters. The cost to be optimized is the validation set error that results from training using these hyperparameters. In simpliﬁed settings where it is feasible to compute the gradient of some differentiable error measure on the validation set with respect to the hyperparameters, we can simply follow this gradient (Bengio et al., 1999; Bengio, 2000; Maclaurin et al.,2015). Unfortunately, in most practical settings, this gradient is unavailable, either because of its high computation and memory cost, or because of hyperparameters that have intrinsically nondifferentiable(不可微) interactions with the validation set error,as in the case of discrete-valued hyperparameters.

To compensate(补偿赔偿) for this lack of a gradient, we can build a model of the validation set error, then propose new hyperparameter guesses by performing optimization within this model.  Most model-based algorithms for hyperparameter search use a Bayesian regression model to estimate both the expected value(预期值) of the validation set error for each hyperparameter and the uncertainty around this expectation. Optimization thus involves a trade-off between exploration (proposing hyperparameters for that there is high uncertainty, which may lead to a large improvement but may also perform poorly) and exploitation(利用) (proposing hyperparameters that the model is confident will perform as well as any hyperparameters it has seen so far—usually hyperparameters that are very similar to ones it has seen before). Contemporary(当代的) approaches to hyperparameter optimization include Spearmint (Snoek et al., 2012),TPE (Bergstra et al., 2011) and SMAC (Hutter et al., 2011).

Currently, we cannot unambiguously recommend Bayesian hyperparameter optimization as an established tool for achieving better deep learning results or for obtaining those results with less effort. Bayesian hyperparameter optimization sometimes performs comparably to human experts, sometimes better, but fails catastrophically(灾难性) on other problems. It may be worth trying to see if it works on a particular problem but is not yet sufficiently mature(成熟) or reliable(可靠). That being said, hyperparameter optimization is an important field of research that, while often driven primarily by the needs of deep learning,
holds the potential to beneﬁt not only the entire ﬁeld of machine learning but also the discipline of engineering in general.  

One drawback common(共同缺点) to most hyperparameter optimization algorithms with more sophistication than random search is that they require for a training experiment to run to completion before they are able to extract any information from the experiment. This is much less efficient, in the sense of(从某种意义上说) how much information can be gleaned(捡来) early in an experiment, than manual search by a human practitioner, since one can usually tell early on if some set of hyperparameters is completely pathological(病态的).  Swersky et al. (2014) have introduced an early version of an algorithm that maintains a set of multiple experiments. At various time points, the hyperparameter optimization algorithm can choose to begin a new experiment, to “freeze” a running experiment that is not promising(前途，有希望的), or to “thaw(解冻)”and resume(恢复) an experiment that was earlier frozen but now appears promising given more information.   

### 11.5 Debugging Strategies

When a machine learning system performs poorly, it is usually difficult to tell whether the poor performance is intrinsic to the algorithm itself or whether there is a bug in the implementation of the algorithm. Machine learning systems are difficult to debug for various reasons.

In most cases, we do not know a priori what the intended behavior(预期行为) of the algorithm is. In fact, the entire point of using machine learning is that it will discover useful behavior that we were not able to specify ourselves. If we train a neural network on a new classiﬁcation task and it achieves 5 percent test error, we have no straightforward way of knowing if this is the expected behavior or suboptimal behavior.   

A further difficulty is that most machine learning models have multiple parts that are each adaptive. If one part is broken, the other parts can adapt and still achieve roughly acceptable performance. For example, suppose that we are training a neural net with several layers parametrized by weights ![](https://latex.codecogs.com/png.image?\dpi{110}%20W) and biases ![](https://latex.codecogs.com/png.image?\dpi{110}%20b). Suppose further that we have manually implemented the gradient descent rule for each parameter separately,  and we made an error in the update for the biases:     
![](https://latex.codecogs.com/png.image?\dpi{110}%20b\leftarrow%20b-\alpha)   
where ![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha) is the learning rate. This erroneous update does not use the gradient at all. It causes the biases to constantly become negative throughout learning, which is clearly not a correct implementation of any reasonable learning algorithm. The bug may not be apparent just from examining the output of the model though. Depending on the distribution of the input, the weights may be able to adapt to compensate(补偿) for the negative biases.   

Most debugging strategies for neural nets are designed to get around one or both of these two difficulties. Either we design a case that is so simple that the correct behavior actually can be predicted, or we design a test that exercises one part of the neural net implementation in isolation(孤立的).

Some important debugging tests include the following.

*Visualize the model in action:* When training a model to detect objects in images, view some images with the detections proposed by the model displayed superimposed(叠加) on the image. When training a generative model of speech, listen to some of the speech samples it produces. This may seem obvious, but it is easy to fall into the practice of looking only at quantitative performance measurements like accuracy or log-likelihood. Directly observing the machine learning model performing its task will help to determine whether the quantitative performance numbers it achieves seem reasonable. Evaluation bugs can be some of the most devastating(破坏性) bugs because they can mislead you into believing your system is performing well when it is not.

*Visualize the worst mistakes:* Most models are able to output some sort of conﬁdence measure for the task they perform. For example, classiﬁers based on a softmax output layer assign a probability to each class. The probability assigned to the most likely class thus gives an estimate of the confidence the model has in its classiﬁcation decision. ypically, maximum likelihood training results in these values being overestimates(高估) rather than accurate probabilities of correct prediction, but they are somewhat useful in the sense that examples that are actually less likely to be correctly labeled receive smaller probabilities under the model. By viewing the training set examples that are the hardest to model correctly, one can often discover problems with the way the data have been preprocessed or labeled. For example, the Street View transcription system originally had a problem where the address number detection system would crop the image too tightly and omit some digits. The transcription network then assigned very low probability to the correct answer on these images.  Sorting the images to identify the most conﬁdent mistakes showed that there was a systematic problem with the cropping. Modifying the detection system to crop much wider images resulted in much better performance of the overall system, even though the transcription network needed to be able to process greater variation in the position and scale of the address numbers.

*Reason about software using training and test error:* It is often difficult to determine whether the underlying software(底层软件) is correctly implemented. Some clues can be obtained from the training and test errors. If training error is low but test error is high, then it is likely that that the training procedure works correctly, and the model is overfitting for fundamental algorithmic reasons. An alternative possibility is that the test error is measured incorrectly because of a problem with saving the model after training then reloading it for test set evaluation, or because the test data was prepared differently from the training data. If both training and test errors are high, then it is difficult to determine whether there is a software defect(缺陷) or whether the model is underfitting due to fundamental algorithmic reasons. This scenario requires further tests, described next.

*Fit a tiny dataset*: If you have high error on the training set, determine whether it is due to genuine(真正的) underﬁtting or due to a software defect. Usually even small models can be guaranteed to be able ﬁt a sufficiently small dataset. For example, a classiﬁcation dataset with only one example can be fit just by setting the biases of the output layer correctly. Usually if you cannot train a classifier to correctly label a single example, an autoencoder to successfully reproduce a single example with high fidelity(保真), or a generative model to consistently emit(发射) samples resembling(类似) a single example, there is a software defect preventing successful optimization on the training set. This test can be extended to a small dataset with few examples.

*Compare back-propagated derivatives to numerical derivatives:* If you are using a software framework that requires you to implement your own gradient computations, or if you are adding a new operation to a differentiation library and must define its `bprop` method, then a common source of error is implementing this gradient expression incorrectly. One way to verify that these derivatives are correct is to compare the derivatives computed by your implementation of automatic differentiation to the derivatives computed by **finite differences**. Because   
![](https://latex.codecogs.com/png.image?\dpi{110}%20f%27(x)=lim_{\epsilon%20\to%200}\frac{f(x+\epsilon)-f(x)}{\epsilon})     
we can approximate the derivative by using a small, finite ![](https://latex.codecogs.com/png.image?\dpi{110}%20\epsilon):     
![](https://latex.codecogs.com/png.image?\dpi{110}%20f%27(x)\approx%20\frac{f(x+\epsilon)-f(x)}{\epsilon})    
We can improve the accuracy of the approximation by using the centered difference:    
![](https://latex.codecogs.com/png.image?\dpi{110}%20f%27(x)\approx%20\frac{f(x+\frac{1}{2}\epsilon)-f(x-\frac{1}{2}\epsilon)}{\epsilon})     
The perturbation size ![](https://latex.codecogs.com/png.image?\dpi{110}%20\epsilon) must be large enough to ensure that the perturbation is not rounded(舍入) down too much by ﬁnite-precision numerical computations.

Usually, we will want to test the gradient or Jacobian of a vector-valued function ![](https://latex.codecogs.com/png.image?\dpi{110}%20g:\mathbb{R}^m\to%20\mathbb{R}^n). Unfortunately, finite differencing only allows us to take a single derivative at a time. We can either run finite differencing ![](https://latex.codecogs.com/png.image?\dpi{110}%20mn) times to evaluate all the partial derivatives of ![](https://latex.codecogs.com/png.image?\dpi{110}%20g),

















