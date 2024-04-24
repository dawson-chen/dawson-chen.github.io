+++
title = 'Deepspeed-HybridEngine开发指南'
date = 2024-01-07T12:37:45+08:00
draft = false
math = true
+++

* *2023-11-29写；*
* *2023-12-06修改：增加适配模型开发流程说明；增加bug解决记录；*

Deepspeed-Chat是一个优秀且易用的PPO开源实现，实际在使用时HybridEngine开发是PPO工程相关的重要一环，本次分享的目的：

*   了解整体Deepspeed的架构，和代码逻辑；
*   清楚如何在Deepspeed上进行HE适配相关开发；
    *   主要是对新的模型结构的适配；

先回答为什么需要做适配，因为HE(hybrid engine)本身解决的问题是将训练中的zero3模型，转换成更高效的对设备通讯压力不大的推理模式，可以是不带tp的全参数推理，也可以是带tp的推理。所以不管是哪种形式的推理，都需要重构推理图，并且处理好这种模式转换间设计的大量引用，也是适配需要做的全部事情。虽然我相信这个过程一定可以改成完全自动的形式，但是目前还没有找到这种实现方式。

然后看一下HE的优势是什么，推理速度不用说，就相当于带或不带zero3的差距，通常10x左右。还有一个优势是带TP的HE方式在内存上的优势，这一点在模型比较大且显存压力较大的场景下尤为重要。下面举ppo的例子，如果训练的sft和actor模型大小是70b，reward和critic是7b，一共4个模型，考虑32张A100-80G的场景：

*   ZeRO3训练模型占用显存（每张卡）：

    Actor: (优化器12 + 参数2 + 梯度2) \* 70 / 32 = 35G
    SFT：通常offload，显存可以记作0G
    reward+critic：3.9G
    每张卡总共需要：38.9G

    占用的显存是够的，但是zero3推理速度在ppo生成阶段几乎不可用；
*   （HE模式）ZeRO3训练模型+TP推理占用显存（每张卡）：

    TP的size设为8

    *   训练阶段：TP的参数释放掉，和ZeRO3模式一样，38.9G；
    *   生成阶段：70 \* 2 = 140G（全部推理参数），TP参数切片140 / 8 = 17.5G，所以一共需要56.4G；

    训练用ZeRO3节约显存，生成用HE提升速度。如果不用TP的HE那么140G放到一张卡上，目前还没有设备能支持。

## 1. 整体架构

### 1.1 启动流程

涉及代码都在deepspeed/launcher目录下面，一共2个文件 runner和launcher。


![image.png](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/734b04c4a96c410fa9e942080b574297%7Etplv-k3u1fbpfcp-jj-mark%3A0%3A0%3A0%3A0%3Aq75.png)

### 1.2 Zero3

*   [x] zero3架构是什么样的，如何实现？

    <https://github.com/microsoft/DeepSpeed/blob/2c2a7f31bcc20ae12ce8d2b8af14448939ebdf12/deepspeed/runtime/zero/stage3.py#L120C9-L120C9>

    自动对参数进行all\_gather，使用后自动释放；实现核心 `ZeROOrderedDict`

### 1.3 Hybrid Engine

*   [x] 为什么需要HE？

    Zero3是用来训练的并行方式，推理的时候有很大劣势，并且不能扩展到多机的情况。
*   [x] HE如何起到作用？实现方式？

    HE内容：1. 自定义算子；2. tensor parallelism；

    *   [x] 不带TP的方式； 适用于 7b、13b  单机多卡
    *   [x] 带TP的方式； 适用于66b、70b 更大模型的训练

    比如说模型ChatGLM2，它的代码实现里transformer块对应的实现类是 `GLMBlock`，HE就是实现一个新的推理过程，带或者不带TP区别就是这个推理过程是不是分布式，然后替换掉 `GLMBlock`的forward方法。

    这种替换方式就像下面这个例子：

    ```python
    class Bird:
        def move(self,):
            print(f'i\'m flying')
    
    class Pig:
        def move(self,):
            print(f'i\'m walking')
    
    bird = Bird()
    pig = Pig()
    
    pig.move = bird.move
    pig.move()
    ''' output
    >>> i'm flying
    '''
    ```


    在python里，Pig想要飞的方式是借用Bird的翅膀，而不需要通过继承实现。

#### 运作流程

Container、Inference、Module、ops之间的调用关系；


![image.png](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/2fe6046f1bf14d2e9cb26a71ef2a2820%7Etplv-k3u1fbpfcp-jj-mark%3A0%3A0%3A0%3A0%3Aq75.png)

HE运作流程，有2个部分：

1.  初始化，module到Continer的创建流程；

    1.  policy定义的位置；
        入口：hybrid\_engine.py populate\_all\_inference\_policies
        policy定义文件：deepspeed/module\_inject/replace\_policy.py
        policy和container对应关系：deepspeed/module\_inject/utils.py
    2.  Container创建过程；    
        ![image.png](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/6daedac42b1c4f28b26ca796fd415589%7Etplv-k3u1fbpfcp-jj-mark%3A0%3A0%3A0%3A0%3Aq75.png)
        1.  Create\_module 创建推理图；新的forward函数
        2.  set\_params\_wo\_copy  container将模型变量赋值给计算图；只给引用 不复制
        3.  forward的替换发生在eval方法里；
    3.  计算图的构建；
        1.  保证和原生pytorch的计算过程保持一致；
        2.  尽量使用ds提前定义的cuda算子；
2.  generate的过程；
    
    ![image.png](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/355b14a0493a486098c522d85d597a79%7Etplv-k3u1fbpfcp-jj-mark%3A0%3A0%3A0%3A0%3Aq75.png)

## 2. 如何适配新的模型

### 2.1 关键点

以chatglm2和BlueLM为例，几个关键点：

1.  如何定义一个新的模型；
    参考HE的架构，主要新增policy、container，以及使用对应的ops重构计算图；

    1.  Policy：
        1.  定义原模型中关键的参数变量；
        2.  定义原模型中的参数引用；
        3.  定义要替换的模块，一般为transformer层级模块对应的类；
    2.  Container；
        1.  保存计算图中需要的参数tensor引用；
        2.  生成推理用的module，用module.forward替换掉原模块中的forward；
    3.  重构推理计算图；
        1.  检查算子是否可以复用；
        2.  检查结果是否正确；
2.  TP相关的代码；

    改动代码之前，请阅读[arxiv.org/pdf/1909.08053.pdf](https://arxiv.org/pdf/1909.08053.pdf)论文，了解Tensor Parallelism的大模型分布式训练方式。在HE中只关注前向推理的流程，适配时最主要的是确定好参数切分的方式。

    1.  参数复制，拆分；
    2.  推理时的reduce操作；
3.  计算流程实现；
4.  \*Inference相关的代码；

    chatglm2 66b 8卡推理；

### 2.2 适配开发流程

HE本质上就是用zero3参数收集起来，重新生成一个计算图（container.module.forward函数）复现并替换原pytorch模型的推理结果；

1.  找到要替换的类，policy对应的类；
    一般在要适配的模型定义文件里面；
2.  找到container.module.forward调用入口；
    在替换模型定义文件里，policy对应的类的forward函数的调用入口，注意入口的参数传递和container对应的forward声明对齐；
3.  检查点：

    *   [x] 参数引用是否正确；
    *   [x] 计算流程是否一致；
4.  单步调试；
    每一步的计算和pytorch代码是否一致；

*   [x] TP代码开发的区别：
    *   container需要单独保存参数；
        原始参数的切片；
    *   在生成阶段，需要先准备tp参数；更重要的是，在生成结束之后要及时释放tp的参数；
        在container的有相应的方法需要实现，主要有参数准备、参数释放；
    *   计算阶段需要适配tp流程；
        显然的变化是，计算用的weight的大小会改变；
    *   注意：generate阶段需要对输入进行拼接；

### 2.3 单元测试方法

*   [x] 如何进行多机程序的debug；

    主要难点是多机程序的调试，这里提供2种方式；

    1.  可以使用pdb在单节点上进行调试代码；代码如下：

        ```python
        ## 在需要阻塞的位置插入下面的代码
        if dist.get_rank() == 0:   # 判断是否为0的进程
            breakpoint()   # 断点
        dist.barrier()  # 当0进程阻塞时，其他节点进行等待
        ```
    2.  （理论可行，未做尝试）用单进程模拟多进程；

        deepspeed启动命令中提供了 `--force_multi`参数，使单进程可以用来模拟多进程，并在vscode中配置启动命令用deepspeed，这样可以使用vscode进行单步调试。

    pdb的了解和相关命令查看[pdb --- Python 的调试器 — Python 3.12.0 文档](https://docs.python.org/zh-cn/3/library/pdb.html)。
*   [x] 调试程序如何写；

    参考deepspeed/tests/下面对应的文件，比如：test\_ds\_bluelm.py。



## 3. 开发中BUGs解决过程

### INFLIGHT状态异常

*耗时：2天*

*   [x] INFLIGHT状态参数出现异常，解决思路：

    *   [x] 验证inflight状态出现的时机；

        *   [x] 分别在生成训练之后打印参数的状态；

            ```text
            第1次生成...
            Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 925})
            第1次生成...
            第1次训练...
            Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 762, <ZeroParamStatus.AVAILABLE: 1>: 163})
            第1次训练...
            第2次生成...
            Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 758, <ZeroParamStatus.AVAILABLE: 1>: 163, <ZeroParamStatus.INFLIGHT: 3>: 4})
            发生报错
            ```

            流程如上面的记录，在第二次训练的时候有参数没有释放；
        *   [x] 查看训练过程中，AVAILABLE状态的变量有哪些？

            *   [x] Available 变量：

                ```python
                'model.layers.x.self_attn.k_proj.lora_left_weight',
                 'model.layers.x.self_attn.v_proj.lora_left_weight',
                 'model.layers.x.input_layernorm.weight',
                 'model.layers.x.post_attention_layernorm.weight'
                40层 * 4 = 160个 
                 'model.norm.weight',
                 'model.embed_layer_norm.weight',
                 'model.embed_layer_norm.bias'
                3个
                ```
            *   [x] Inflight 变量

                ```python
                'model.layers.39.self_attn.q_proj.weight',
                 'model.layers.39.self_attn.q_proj.lora_right_weight',
                 'model.layers.39.self_attn.q_proj.lora_left_weight',
                 'model.layers.39.self_attn.k_proj.weight'
                ```

            大概率和lora是有关系的，从这里下手找原因；
        *   [x] 修复container中lora参数的bug；
        *   [x] 修复mlp参数中lora没有起作用；
        *   [x] 修复mlp和attention中参数拷贝的bug；

            *   [x] BlueLMAttention细节：
                *   self.attn\_qkvw 需要等于None；
                *   self.attn\_qw, self.attn\_kw, self.attn\_vw 在推理过程中合并成qkvw；
                    使用提前开辟好内存的方式

        \==修复lora相关代码后，问题仍然存在。==
    *   [x] 测试glm2-6b用zero3训练的时候，参数状态变化情况；

        ```python
        生成阶段
        Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 423})
        
        训练阶段
        Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 338, <ZeroParamStatus.AVAILABLE: 1>: 85})
        
        训练后第一次生成阶段
        Counter({<ZeroParamStatus.NOT_AVAILABLE: 2>: 337, <ZeroParamStatus.AVAILABLE: 1>: 85, <ZeroParamStatus.INFLIGHT: 3>: 1})
        ['transformer.encoder.layers.0.input_layernorm.weight', 'transformer.encoder.layers.0.self_attention.query_key_value.bias', 'transformer.encoder.layers.0.post_attention_layernorm.weight', 
         ... # 每一层
         'transformer.encoder.final_layernorm.weight']
        ['transformer.encoder.layers.27.mlp.dense_4h_to_h.weight']
        ```

        和bluelm测试的效果差不多，但是glm6b可以正常训练；

        *   [x] 检查glm6b的代码有什么区别；

            *   训练脚本； <font style="color:green">gradient checkpoint、offload不一致</font>
            *   container 参数复制；

            \==去掉gradient checkpoint之后可以正常训练；==
    *   [ ] ~~梳理zero3过程，找到为什么会出现inflight状态，以及对应的意义；~~

        *   [ ] ~~AVAILABLE 和 INFLIGHT 状态转换图~~
        *   [ ] ~~INFLIGHT # parameters are being gathered.~~


### 显存超出bug

*耗时：一天*

*   [x] 显存超出BUG解决；

    *   [x] 原因分析：显存workspace申请不够，导致在使用算子计算softmax的时候超出限制；
    *   [x] 解决方法如下：
        1.  修改自定义算子申请显存的逻辑；
            代码在csrc/transformer/inference/includes/inference\_context.h#L98 GenWorkSpace函数里
        2.  ~~调大ds\_bluelm.py里面申请显存的参数；~~
            ~~代码在deepspeed/model\_implementations/transformers/ds\_bluelm.py~~
        3.  BlueLMAttention里forward计算attention的部分改成pytorch实现；
            pytorch动态申请显存，只要显存够就不存在溢出的风险；
*   [x] 第2种方法，通过调大长度参数，仍然没有解决；
*   [x] 第3种方法，使用pytorch重新定义attention计算过程；

    *   [x] kv-cache适配时，遇到present\_key\_value向量无理由自发改变的问题；

        
        ![image.png](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/1a825d9e1be5458c9ee0d59295d550cc%7Etplv-k3u1fbpfcp-jj-mark%3A0%3A0%3A0%3A0%3Aq75.png)

        通过改变present\_key\_value内存地址解决；==原因未知，如果有，就是对cuda敬畏之心不够！==

