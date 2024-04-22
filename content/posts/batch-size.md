+++
title = 'Batch Size杂谈'
date = 2024-01-22T23:36:14+08:00
draft = false
math = true
+++

在OpenAI 2018年的一篇论文《An Empirical Model of Large-Batch Training》中就介绍了batch size的选择问题，论文中gradient noise scale作为选择batch size的关键指标。

简单来说，我们实际中用的SGD产生的loss是真实loss的一种近似，这种近似对应的偏差和我们选择的batch size相关，batch size越大偏差越小。究其本质原因，近似的偏差与批数据的信息有关，当训练loss比较大的时候 可以认为数据中的所有信息都是相关信息，而当loss越来越小，批数据中包含的信息偏差占比会越来越高。

论文中最大的亮点在于通过严密的推论得出上面的结论，并且推导出固定的模型要达到相同的loss，在不同的batch size下所需训练时长和需要的训练数据量之间的关系，如下所示：
$$
(\frac{S}{S_{min}}-1)(\frac{E}{E_{min}}-1)=1 \tag{A1}
$$
其中$E=BS$表示训练使用的数据，通过这个公式，可以得到：

- batch size越大，训练step减小，所需的数据会增加；
- 训练所需的步数有最小值；

分别解释一下，第一点是因为：当数据中有用信息占比高得时候，小batch size和大batch size得到得梯度方向是差不多的，因此数据的使用率就比较低。第二点，同样是数据的利用效率，如果把batch size很大的情况下，gradient noise已经很小，继续将batch size翻倍得到的收益就很小，所以即使batch size增加到很大，依然需要一定数量的更新步数，也就是$S_{min}$。

论文中基于gradient noise scale给出一个batch size的经验选择是$B=E_{min}/S_{min}$，在OpenAI Scaling Laws论文中进一步根据经验总结为：
$$
B_{crit}\approx \frac{B_*}{L^{1/\alpha_B }} \tag{3}
$$
其中，$B_*$和$\alpha_B$为经验拟合值，分别约等于$2\times10^8$和$0.21$。

**问题：为什么梯度累积太大会导致loss升高？**

如果batch size远小于$B_{crit}$那么训练的时间会大大增加，但是并不会显著的减小需要的训练量。需要注意的是这里假设了无限的并行条件，当我们在实际中使用梯度累积增大batch size，使得更接近$B_{crit}$​，那么训练总步数会减少，但是总的时间反而会增加。


下面进行说明，根据公式A1可以得到：
$$
E=\frac{E_{min}}{1-\frac{S_{min}}{S}}
$$
意味着，当我通过梯度累积增加batch size，S会减小 但是为了达到同样loss所需的训练数据会增加。而梯度累积并不影响训练速度，过相同的case需要的时间是一样的，也就是需要更多的时间才可以达到同样的loss。

这个结论也告诉我们，如果只是为了提升训练速度或者提升训练效果，梯度累积并不会有帮助。当然还由其他的影响因素，比如PP并行方式需要大的batch size提升计算效率，或者提升算法训练的稳定性。

#### 不同batch size下数据的修正

由公式A1得到：
$$
D_{min} = \frac{D}{1+\frac{B}{B_{crit}}}
$$
如果2个不同的batch size，达到相同的loss所需数据量的比值为：
$$
\frac{D_1}{D_2} = \frac{B_{crit}+B_1}{B_{crit}+B_2}
$$
通过这个修正公式可以看到，如果选择了不正确的batch size，那么会导致实际训练的token的作用并没有最大化。所以，可能会出现实际训练1.2T的token，但实际上仅相当于500B的效果。


> 欢迎关注我的个人公众号：AI学舍
>