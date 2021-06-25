In word2vec, the conditional probability distribution is given by taking vector dot-products and applying
the softmax function:

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(O=o|C%20=%20c)=\frac{exp(u^T_0v_0)}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)})   

Here,  ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_0) is the ‘outside’ vector representing outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20o), and ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c) is the ‘center’ vector representing center word ![](https://latex.codecogs.com/png.image?\dpi{110}%20c). To contain these parameters, we have two matrices, ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) . The columns of ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) are all the ‘outside’ vectors ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_w). The columns of ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) are all of the ‘center’ vectors ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_w). Both ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) contain a vector for every
![](https://latex.codecogs.com/png.image?\dpi{110}%20w%20\in%20Vocabulary) .   
Recall from lectures that, for a single pair of words c and o, the loss is given by:  

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,o,U)=-logP(O=o|C=c))&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;  (2)  

Another way to view this loss is as the cross-entropy between the true distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and the predicted
distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) Here, both ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) are vectors with length equal to the number of words in the vocabulary.Furthermore, the ![](https://latex.codecogs.com/png.image?\dpi{110}%20k^{th}) entry in these vectors indicates the conditional probability of the ![](https://latex.codecogs.com/png.image?\dpi{110}%20k^{th}) word being an ‘outside word’ for the given c. The true empirical distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) is a one-hot vector with a 1 for the true outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20o) , and 0 everywhere else. The predicted distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) is the probability distribution  ![](https://latex.codecogs.com/png.image?\dpi{110}%20P(O|C=c))given by our model in equation (1).  
(a) (3 points) Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss
between ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) ; i.e.,show that 

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{w%20\in{Vocab}}y_w%20log(\hat{y}_w)=-log(\hat{y_o}))&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &ensp;&ensp;&ensp;&ensp; &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(3)   

Your answer should be one line.  
Answer:  

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{w%20\in{Vocab}}y_w%20log(\hat{y}_w)=-y_olog(\hat{y_o})-\sum_{w%20\in%20Vocab,w\neq%20o}y_wlog(\hat{y_w})=-log(\hat{y_o}))
 
(b) (5 points) Compute the partial derivative of &ensp; ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,%20o,%20U)) &ensp; with respect to ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c) . Please write your answer in terms of ![](https://latex.codecogs.com/png.image?\dpi{110}%20y,\hat{y}) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c).  

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,o,U)=-log(P(O=o|C=c)))      
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-log(\frac{exp(u_o^Tv_c)}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)})=-u_o^Tv_c+log(\sum_{w\in%20Vocab}exp(u_w^T%20v_c)))
![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J_{naive-softmax(v_c,o,U)}}{\partial%20v_c}=-u_o+\frac{\partial%20(log(\sum_{w\in%20Vocab}exp(u_w^T%20v_c)))}{\partial%20v_c})     
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o+\frac{1}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)}\sum_{w\in%20Vocab}\frac{\partial%20(exp(u_w^T%20v_c))}{\partial%20v_c})        
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-u_o+\frac{1}{\sum_{w\in%20Vocab}exp(u_w^T%20v_c)}\sum_{w\in%20Vocab}exp(u_w^T%20v_c)u_w)      
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-u_o+\sum_{w\in%20Vocab}P(O=w|C=c)u_w)       
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-u_o+\sum_{w\in%20Vocab}\hat{y_w}u_w)  ---(y_hat - y)U  

(c) (5 points) Compute the partial derivatives of ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,%20o,%20U)) with respect to each of the ‘outside’ word vectors, ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_w)’s. There will be two cases: when ![](https://latex.codecogs.com/png.image?\dpi{110}%20w=o), the true ‘outside’ word vector, and ![](https://latex.codecogs.com/png.image?\dpi{110}%20w\neq%20o), for all other words. Please write you answer in terms of ![](https://latex.codecogs.com/png.image?\dpi{110}%20y,\hat{y}) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20U)

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J(v_c,o,U)}{\partial%20u_w}==\frac{\partial%20u_o^Tv_c}{\partial%20u_w}+\frac{\partial%20(log\sum_{w\in%20Vocab}exp(u_w^Tv_c))}{\partial%20u_w})   
当 ![](https://latex.codecogs.com/png.image?\dpi{110}%20w\neq%20o) 时     
![](https://latex.codecogs.com/png.image?\dpi{110}%20-\frac{\partial%20u_o^Tv_c}{\partial%20u_w}=0)    
![](https://latex.codecogs.com/png.image?\dpi{110}%20\therefore%20\frac{\partial%20J}{\partial%20u_w}=\frac{\partial%20(log\sum_{w%20\in%20Vocab}exp(u_w^Tv_c))}{\partial%20u_w}=\frac{1}{\sum_{w%20\in%20Vocab}exp(u_w^Tv_c)}\sum_{w%20\in%20Vocab}exp(u_w^Tv_c)v_c)     

