+++
title = 'Megatron-LM解读：流水线并行原理和代码解读'
date = 2024-02-05T23:34:01+08:00
draft = false
math = true
+++


Megatron中包含了大多数目前大模型预训练中所需要的并行技术，并且相较于Deepspeed在硬件层面可以得到更多的优化支持。Megatron的优势体现在其先进的并行化设计上面，而其中流水线并行是非常重要的创新点。相比于tensor并行，流水线并行对通讯的压力更小，使得多机训练超大模型成为可能。而在现代工业生产体系里，流水线早已经是一种耳熟能详的科学管理方式。本文中我们结合工业流水线的视角，分析megatron中流水线的设计与具体实现方式。

在大模型蓬勃发展的时代，超大模型训练对框架能力的要求越来越高，工作分工也逐渐演变成算法+框架工程师2部分合作的模式。虽然说专业的人干专业的事情，但是算法工程师对框架原理有一定了解依然是必要的。这里有3个我认为重要的理由：

- 模型训练中遇到的大多数的问题是算法和工程的混合问题；
- 算法工程师需要能够独立开发一些小需求；
- 有助于和框架工程师沟通效率的提升；

## 认识流水线

除了曾经在流水线上短暂的工作经验之外，我对流水线的认识主要来自于一个汽车工业发展早期的故事。

1903年的美国底特律，一位出生自普通家庭的汽车工程师亨利福特开始了他的第二次创业。福特拥有多年的汽车设计经验，在上次创业过程中他制造出了性能出众的汽车，虽然通过早期的赛事证明了自己，但最终因为汽车造价太高而导致无法被大众接受，从而导致公司破产。这次福特吸取经验打算将目光瞄向民用市场，制造可以被广大的农场主接受的汽车。

通过多年快速的技术迭代后，在1908年福特推出了后来鼎鼎大名的T型车，出色的设计以及过硬的质量使得该车很快成为了市场上的抢手货。到了1913年，福特已经是美国最出名的企业家之一，个人财富更是达到了数十亿。此时，可以说是名利双收的福特却开心不起来，一方面T型车的订单已经排到了几个月以后，另一方面工厂的生产效率却一直提不上去。为了实现当初定下的目标，福特目前最关心的只有2件事情，一是继续扩大产能，二是降低汽车的成本。放到今天的工业生产模式下，扩大生产几乎就等于降低产品成本，但在当时的手工作坊装配模式下，扩大生产意味着人才需求的直线上升以及巨大的管理成本，福特知道生产模式的变革已经迫在眉睫。

1913年福特引入了流水线作为装配生产的形式，从最初的车架被推上流水线，经过各个部件的装配，最终被推下流水线，全流程耗时不过几个小时。效率的提升不光将产能提升数倍，同时汽车的成本下降到原来的一半，也为后来福特推出双休制拉高工人工资奠定了基础。毫无疑问，这是福特汽车发展史上最辉煌的时代。

<img src="https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/DALL·E 2024-02-05 14.19.30 - A historical depiction of Ford's assembly line from 1913, capturing the momentous introduction of mass production techniques for automobile assembly. .webp" alt="" style="zoom: 50%;" />

<center><i>福特公司在使用流水线装配T型汽车的场景 from DALL·E</i></center>

为什么一个简单的流水线能够带来这么大的改变，我们分析一下流水线的特点，首先是将原来复杂的工作流程切分成多个小块，每个工人只需要负责其中某个单子的工序，降低了对装配工人技术的依赖；另外，不同于传统的扩大生产需要同时扩大生产设备的模式，流水线并不需要设备的增加，保证了成本降低；最后也是最重要的，流水线子程序划分的越细小，生产效率的提升越高。

为什么要讲这个故事，因为在Megatron中使用的流水线并行的思想与工业生产中的流水线不能说一模一样，只能说是完全一致。对传统流水线有基本的认知可以帮助理解流水线并行，另外哪有那么多创新啊，说白了就是互相借鉴罢了。

## 流水线并行设计

*流水线并行的英文是Pipline Parallelism，后面简称PP。*

生产生产中的流水线是把一个复杂的过程拆分成多个简单的子过程，每个工人只负责其中的一部分。在模型推理中的PP与传统流水线是完全一致的，大模型被按照层数拆成多个子模型，由单独的GPU设备负责子模型的推理计算，每个节点做的事情就是不断的接收前面节点的输入，然后计算完并将结果传给后面的节点。整个流水线的吞吐量与流水线的细分程度相关，并成线性关系（忽略切分的不均匀度，以及进入时间和流出时间）。

![image-20240204143433737](https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/image-20240204143433737.png)

当PP用在训练过程中的时候，事情发生了一点变化。因为训练是一个双向的过程，所以整个流水线会变成如上图所示。此时流水线的进入节点和流出节点是同一个，或者可以理解为每个节点同时处在2条方向相反的流水线中。此时，每个节点执行前向动作与后向动作的序列方式就叫做流水线策略。下面从简单到复杂介绍3种不同的流水线策略，并通过简单的语言说明设计的原理。

### Fill-Drain

Fill-Drain就是先运行前向流水线，然后再执行反向流水线。整体的流程如下图所示：

![image-20240204125208806](https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/image-20240204125208806.png)

通过类比生产流水线，我们很好理解当流水线开始运行的时候，处于流水线后面的节点会有一段等待时间。所以出于利用率考虑，我们在使用流水线的时候，希望执行的长度越多越好。在PP里面，就是我们希望在前向流水线执行足够多的批次之后再开始执行反向流水线，从而降低设备空闲的时间，在这个图里面也可以叫做bubble time。

然而，事情并没有那么顺利，我们知道模型在前向过程中需要记录计算的激活值，用来在反向传播的时候计算对当前节点输入的梯度。所以节点的显存上限决定了序列长度的最大值，因此引出了1F1B的流水线策略。

### One Forward and One Backward(1F1B)

显然，如果降低了激活值带来的显存消耗，就可以尽可能的增加执行的序列长度。因为前向流水线执行过程中，输入是分批次的，每个批次对应的梯度是可以单独计算的，所以第1份输入对应的梯度并不需要等所有批次都执行完后才开始计算。由此可见，在流水线中需要尽量提前反向传播的时间是一种有效的方式。每个批次梯度计算最早的时间节点就是在最后一个前向流水线节点执行完之后，因此形成的策略就是流水线中的每个节点在执行完前向之后，只要有反向流水线的任务，就需要执行一次反向流水线的任务。

![image-20240204125227494](https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/image-20240204125227494.png)

最终形成的序列执行顺序如上图所示，这种方式相对于Fill-Drain并没有减少气泡的时间，但是因为降低了激活值占用的显存，因此可以使用更长的序列长度，从而增加了设备的利用率。

### Interleaved 1F1B

我们在与传统流水线的类比中解释过，流水线切分的越细，整体的吞吐量越高。但是当设备数量一定的情况下，有没有办法能够增加流水线切分的粒度呢，那就是Interleaved 1F1B。

![image-20240204125245015](https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/image-20240204125245015.png)

如上图所示，这种方式是将流水线划分的更细，但是因为设备数量是固定的，所以每个节点上需要执行多个子模型，最终流水线执行如下图所示：

![image-20240204125303310](https://raw.githubusercontent.com/AllenChennn/picgo-repo/master/image-20240204125303310.png)

事实上，这种情况下每个设备节点同时处于4条流水线中，2条前向流水线，2条后向流水线。这种方式可以减少bubble time，从而提升了设备利用率。

## 代码实现

其实只要理解PP设计的原理，那么代码写起来也是非常清晰的。我们先用伪代码描述整个流程，然后分析一下megatron实现过程中的要点。megatron里主要使用的是1F1B的调度方法，因此我们以1F1B为例进行分析，其他实现类似。

虽然流水线调度看起来很复杂，但其特点是每个节点的操作逻辑基本是一致的，边界情况也很清晰，所以实现的时候只需要考虑每个节点的执行逻辑即可。

### **节点间通讯**

以传统的流水线为例，假设我们需要用代码实现一个多进程的流水线。那么要素一共有3个：

- 等待上一个节点的执行结果；
- 执行当前节点的操作；
- 将结果传给下一个节点；

PP实现的时候，因为每个节点对应了前向后向2条流水线，所以需要4种通讯能力，分别对应了前后2条流水上接受和发送的能力。megatron在实现的时候有一个专门的函数用来实现这种功能，下面是源码：

```python
def communicate(tensor_send_next, tensor_send_prev, recv_forward, recv_backward):
    """Communicate tensors between stages."""
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float
    if recv_forward:
        tensor_recv_prev = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_backward:
        tensor_recv_next = torch.empty(tensor_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Send tensors in both the forward and backward directions as appropriate.
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_prev,
                                               mpu.get_pipeline_model_parallel_prev_rank())
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_prev,
                                               mpu.get_pipeline_model_parallel_prev_rank())
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_next,
                                               mpu.get_pipeline_model_parallel_next_rank())
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_next,
                                               mpu.get_pipeline_model_parallel_next_rank())
        ops.append(recv_next_op)
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    return tensor_recv_prev, tensor_recv_next
```

`torch.distributed.P2POp`是点对点操作的封装类，`torch.distributed.isend`是要执行的异步操作，操作执行发生在`torch.distributed.batch_isend_irecv`里。

### 调度流程伪代码

下面为1F1B的整体流程，megatron在实现的时候将warmup和cooldown单独拆分出来，其他逻辑大体是一致的，感兴趣的可以参照着去看源代码。

```python

def forward_backward_step(losses):
    ## 1F1B
    # forward stage
    if not cooldown_stage():
        if not is_first_node():
            fwd_inputs = wait recieve_forward_pipline()
        else:
            fwd_inputs = next(data_loader)
        fwd_outputs = model.inference(fwd_inputs)
        wait send_forward_pipline(fwd_outputs)

    # backward stage
    if not warmup_stage():
        if is_last_node():
            loss = fwd_outputs
            losses.append(loss)
            bwd_outputs = optimizer.backward(model, loss)
            wait send_backward_outputs(bwd_outputs)
        else:
            bwd_inputs = wait recieve_backward_outputs()
            bwd_outputs = optimizer.backward(model, bwd_inputs)
            if not is_first_node():
                wait send_backward_outputs(bwd_outputs)

losses = []
while len(losses) != num_micro_batches:
    forward_backward_step(losses)

optimizer.step()
```

推荐大家刚开始学习megatron-lm的时候去看一下v2.0的版本，包含了pipline parallelism，tensor parallelism，data parallelism，fp16混合精度训练 的特性，一共1w行代码比较简单易懂。

