+++
title = '混合精度训练'
date = 2023-07-05T22:05:35+08:00
draft = false
math = true
+++

神经网络的参数是用浮点精度表示的， 浮点精度的标准是[IEEE 754 - Wikipedia](https://en.wikipedia.org/wiki/IEEE_754)，以下是一个FP16数值在内存中存储格式。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/69dad0f777664397b6fd63a0e51ce01b~tplv-k3u1fbpfcp-watermark.image?)

随着神经网络模型规模越来越大，如何减少模型占用的内存并且缩短训练时间成为亟需解决的问题，混合精度训练就是其中之一的解决方案，并且几乎不会影响模型训练的效果。

### 混合精度原理

想象一下，如果模型参数+loss+gradient都是用fp16保存的，fp16的最小值是$6.1\times 10^{-5}$，小于最小值的gradient都会变成0，相当于浪费了一次梯度传播。或许小的gradient并没有很重要，但是积累多次就会变得不可忽略。当前大模型普遍较低的学习率也会加剧这个问题的影响。

因此为了解决这个问题，就需要用更高精度fp32保存一份参数，在正常前向推理和反向传播时都用fp16，计算好得梯度先转换为fp32，再乘以学习率，然后更新到fp32存储得参数上，最终将fp32参数转换成fp16更新模型参数。

整个流程如下如：

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/44c5d5434195494497788c6c8696486a~tplv-k3u1fbpfcp-watermark.image?)

这种用fp16和fp32共同训练模型得技术就叫做混合精度训练(MP, Mixed-Precision training)，显然MP并不能节省模型加载需要的内存，因为需要多存储一份fp16的参数和梯度，但是用fp16进行模型前向和后向计算，能够减少中间计算值存储需要的内存，这部分内存会随着sequence length和batch size增大而增大，所以只有在这部分中间值占用内存比重较高时才能带来一定的内存节约。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c804603140bf452380710e1cf0912b2e~tplv-k3u1fbpfcp-watermark.image?)

虽然计算时间的影响不大，但是fp16训练时间的确会大大减少，通常是减少1.5~5.5倍。

> 更多资料：
>
> [fastai - Mixed precision training](https://docs.fast.ai/callback.fp16.html#A-little-bit-of-theory)
>
> [Understanding Mixed Precision Training | by Jonathan Davis | Towards Data Science](https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4)

### Loss Scale

是不是混合精度训练就完全没有梯度损失了呢，并不是，在反向传播过程中其实已经有部分梯度因为精度原因丢失了（因为正常模型梯度都不会太大，所以我们主要考虑下溢出）。那么如何解决这部分问题呢，就要用到Loss Scale。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4e30527b86d24f16b20b39ef4fac41b8~tplv-k3u1fbpfcp-watermark.image?)

原理是将Loss乘以一个比较大的数scale，因为Loss是用fp32存储的，所以scale的选值范围是比较大的。这样因为反向传播链式法则原理，梯度也会放大很多倍，原本下溢出的值也会保存下来。然后在梯度转换成fp32后除以scale，最后更新就与正常混合精度训练一致了。

流程如下：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a7c0424fa10f40ddbb67f202e05f73fd~tplv-k3u1fbpfcp-watermark.image?)

一般在开始训练时scale会设定成一个比较大的值，如果计算过程中fp16梯度发生上溢出，会跳过当前步的参数更新，并将scale下调。训练log中会输出如下消息：

<aside> ⚠️ Gradient overflow. Skipping step, loss scaler 0 reducing loss scale to…

</aside>