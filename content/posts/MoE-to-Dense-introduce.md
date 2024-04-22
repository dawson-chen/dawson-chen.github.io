+++
title = 'MoE to Dense介绍以及相关论文速览'
date = 2024-04-22T23:20:46+08:00
draft = true
math = true
+++


## 背景

MoE模型可以在推理算力不变的情况下，继续扩大模型的规模，从而获得到scaling up带来提升。但是在实际应用场景下，这种提升并非没有代价。

1. 模型的推理性能；
   
    因为MoE训练带来显著的通讯量提升，并且在越大规模上面这种提升越巨大，所以MoE的训练性能相比于同样激活参数量的Dense网络只有50%~80%。但当模型处于真实应用场景下，相比与训练性能，我们更关心的是MoE模型的推理性能，MoE模型的推理性能严重依赖于设备之间的通讯带宽，因此会给部署带来额外的成本。
    
2. 端侧应用的限制；
   
    MoE模型虽然激活参数较少，但是模型的总参数量会增大数倍，这在端侧这种内存受限的场景下应用起来并不容易。虽然，在服务端应用的时候可以通过EP这种方式极大的降低总参数量带来的影响。
    

因此MoE to Dense的技术可以使MoE模型能够克服上面2个缺点（当然了，因为已经变成一个彻底的Dense模型）。并且，考虑到MoE模型expert之间存在极大的冗余性，缩小MoE总参数量就看起来是非常合理的一种需求了。

## 2篇相关论文

## **One Student Knows All Experts Know: From Sparse to Dense**

*National University of Singapore, Huawei, Oct 2022*

**总结：**

> 应该是第一篇提出将MoE能力压缩到Dense中，看得出来Huawei在发展MoE技术上还是走到前面的。同时结合手机业务的应用场景（背景中说的第2点），提出了MoE to Dense的技术。

![aaa](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/moe-to-dense1.png)

文章提出了一项任务knowledge gather，就是将多个expert中的知识合并到单个expert中，以训练出与 MoE 模型效果类似的稠密学生模型。该任务分为知识收集和知识蒸馏两个阶段，知识收集中探索了四种不同的知识收集方法，知识蒸馏则利用整合好的知识进一步优化学生模型。在实验中，该方法在计算机视觉和自然语言处理两个领域取得了优异的表现。

知识收集方法分为4种：summation、averaging、Top-K Knowledge Gathering (Top-KG)、Singular Value Decomposition Knowledge Gathering (SVD-KG)。前2个方法类似于模型的参数合并，而后面2种方法是论文中提出的，可以尽可能把重要的参数提取出来。不管用哪种方法，合并都给参数中引入了噪声，因此下一步就是用蒸馏的方式恢复模型的能力。

> 论文中的主要创新应该是知识收集的方式，那么最终要的应该是验证知识收集的能力，但可惜的是给出的结果并没有充分的验证。MoE to Dense应用很重要的一点是花尽量少的代价将MoE的能力迁移到Dense模型上面，论文中并没有说明第二阶段蒸馏用的计算量，而是从蒸馏后最终效果和传统的蒸馏方法进行对比。
> 

### Experts Weights Averaging: A New General Training Scheme for Vision Transformers

*Aug 2023, Fudan University*

re-parameterization，即二次参数化方法，是在CV中提出的一种方法，旨在解决多分支类型的网络结构在推理时的低效，比如 ResNet。具有代表性的是RepVGG，在训练的时候使用多分支结构，但是在推理阶段使用卷积核合并得到一个单分支的网络。该方法最重要的是合并后的结构等价性，而MoE的expert并不存在等价的合并方式。

因此，论文为了解决这个问题，在每次训练后人为的将expert之间的参数距离拉近。方法如下：

![aaa](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/moe-to-dense2.png)

> 这里的做法可能有一点隐患，因为MoE的训练过程是会导致expert之间的差异越来越大，如果训练中人为对expert之间参数进行了平滑，那么是否同时也降低了MoE能取得的效果呢？
> 

在训练结束后，通过平均每个 MoE 的专家，将每个 MoE 转换为 FFN，将模型转换回原始 ViT 以进行推理。论文还提供了理论分析，证明了该方法的有效性和通用性，并在各种 2D 和 3D 视觉任务、ViT 架构和数据集上进行了广泛实验。

> 这篇文章的出发点是利用MoE结合重参数化提升ViT的效果，同时也降低了MoE模型的部署难度，是一个不错的思路。
> 

## 后记

MoE to Dense并不是一个很常见的需求，2篇论文解决的场景都或多或少都有点推理资源敏感。但我觉得随着MoE的模型越来越大，那么对应的推理压力也会越来越大，虽然有专家并行，但实际要实现和同激活参数的Dense模型相同的推理效率并不容易，因此MoE to Dense也会变得越来越有价值。另外MoE中一定存在大量的冗余信息，可以简单说2个现象：1. 增加激活专家并不会带来明显的效果增益；2. 不管用什么方法训练，在推理的时候有些专家被激活的比例任然比较少，因此对MoE做裁剪是必须得一个步骤，而裁剪和转换Dense都需要搞清楚MoE学习到的参数特性。

这个方向也有很多的挑战，举2个方面：

1. 目前MoE结构趋向于Deepseek提出的Fine-Grained + shared expert 方式，这又给MoE to Dense的转换增加了难度。因为不光要考虑转换方式有效性，同时还要兼顾模型结构的变换。
2. 这个事情有一个不在明面上的好处是，通过验证不同的转换方案同时也得到一些MoE技术内在的insight。但是这个事情再深一点就要考虑模型参数的可解释性，这是一个更加困难的领域。