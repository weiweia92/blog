## 1.Machine Learning and Neural Networks

**(a)** i.Because SGD uses mini-batches and not the entire dataset to calculate the gradient for updating the weights, the updates suffer from a significant amount of noise. Momentum does accumulating of updates over a time horizon into a recent weighted average to update the weights. The trajectory(轨迹) the weights follow through their vector space therefore becomes less jittery(抖动), makes fewer jumps "left or right" and more closely follows the true downward direction of the cost function.

  <ol>ii.
