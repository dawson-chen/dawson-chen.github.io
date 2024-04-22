+++
title = 'ChatGPT Plugins原理介绍和讨论'
date = 2023-04-07T20:43:43+08:00
draft = true
math = true
+++
## 背景

让我们回顾以下过去的半个月里重要的AI发展。

| 事件             | 时间   | 介绍                                  | 公司        |
| -------------- | ---- | ----------------------------------- | --------- |
| Visual ChatGPT | 3-12 | 可以通过文本和图片聊天，甚至修改图片内容。               | Microsoft |
| GPT4发布         | 3-13 | 更大的ChatGPT模型，部分专业能力达到人类水平，可以接收图片输入。 | OpenAI    |
| 365 Copilot    | 3-16 | 智能办公大杀器。                            | Microsoft |
| 文心一言           | 3-16 | 中国版的ChatGPT                         | Baidu     |
| ChatGPT plugin | 3-23 | 可以使用工具的ChatGPT                      | OpenAI    |
| HuggingGPT     | 3-30 | 可以使用HuggingFace中模型能力的ChatGPT        | Microsoft |

很多评价说过去的几周是AI发展的Crazy Week，这种速度疯狂到甚至让人们开始担心AI会影响到社会和人类，并在公开信中呼吁暂停AI的研究。造成这种现象的原因可以理解为，一是基于ChatGPT的成功，二是行业内大量的关注。

个人认为，这其中ChatGPT plugin可以认为是对行业应用最有影响力的一个技术，也是继ChatGPT发布以来OpenAI发布的最重要的更新，可以简单的理解为OpenAI发布了对应ChatGPT的应用商店。对未来人工智能应用的形态也有一定启发，以前的AI模型的定位更多的是充当的一个单一的智能工具，具体到某个任务上，还需要人工协同才能完成；但是有了plugin这项技术，那么AI模型可以代替之前人工的部分，自主使用工具，从而端到端的完成某一项任务。这也是为什么一些基础的工作岗位很有可能会被新一代AI技术取代。

在网上已经有很多对ChatGPT plugin如何使用的介绍，但是比较少有对其实现原理进行解析的内容。这篇文章里我们主要分析一下它的原理，以及可能造成的影响。

### 必要性

首先说为什么语言模型要使用插件？随着语言模型的规模不断变大，各种涌现能力被相继发现，从而衍生出各种关于模型能力的研究。但谈到语言模型的应用，始终绕不开一个问题，就是模型无法获取外界的信息。也就是，一旦模型训练完成，后续的所有输出都来自于训练数据中学习到的知识。


大语言模型存在的问题可以总结为以下2点：

-   缺少最新数据的补充；

    在不同的应用场景，对数据的需求也是不同的。在开放问答领域，可以是维基百科一类的数据。在特定业务领域，可能是公司内部的一些私人数据集。

-   缺少专业的能力；

    大型语言模型对通用逻辑的理解是比较好的，比方说写一篇文章，与人聊天。但是涉及到特殊的专业，比方说作数学题、求公式的解，这类型问题对模型来说是有点难的。

    虽然GPT4号称用了更大的模型，可以在一些专业领域得到类似于人类的效果甚至超越。但是从本质上来看，语言模型所采用的文字接龙训练方式，对于这类问题是非常不友好的。

    或许随着模型变大，训练时间更长可以得到更好的效果，但是花费巨大训练出的GPT3在计算能力上远远达不到1970年代出现的计算器，本身就可以说明大模型技术是不足以解决专业推理问题的。

了解了以上模型存在的问题，就可以理解教模型使用插件的必要性了。PS：使用插件、使用工具，在不同的地方有不同的说法，但是是一件事情。

## 模型使用工具技术发展

在GPT3发布以后，就有一些AI模型使用插件的技术研究陆续出现，甚至有一些开源的框架在github上收获不错的关注。

### **想法的提出：MRKL System**

MRKL System（全名是Modular Reasoning, Knowledge and Language，论文[链接](https://arxiv.org/abs/2205.00445)，博客[链接](https://www.ai21.com/blog/jurassic-x-crossing-the-neuro-symbolic-chasm-with-the-mrkl-system)）由以色列的一家人工智能公司AI21推出，可以被认为是语言模型使用工具系统想法的提出者。虽然在此之前有WebGPT这类教模型使用浏览器的工作，但它是第一个提出将模型作为中枢，接入各种不同类型的插件来完成任务的工作。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c1c9b7cc490641a2a998832ffd0229ab~tplv-k3u1fbpfcp-watermark.image?)

从工作流程上来看，MRKL已经完全接近于ChatGPT plugin。MRKL认为这是一种跨越神经学派和符号学派的架构（neuro-symbolic architecture），各种插件可以被认为是符号系统，由神经学派的语言模型进行统一调用。

这篇论文中以使用计算器为例子，主要描述了如何将自然语言中的内容转换为API所需要的参数，文中提出语言模型few-shot在复杂的问题上性能有限，所以用Prompt tuning这种轻量化的微调技术提升转换的准确率。Prompt tuning技术是用特定训练好的非自然语言prompt来控制模型在特定任务中的生成表现，对应到MRKL中那就是每一个插件都需要训练一个特定的Prompt，虽然说有一定训练成本，但也算是一种比较好的解决思路。

可是文中对于最重要的问题：”怎么决定调用插件？“，这块的细节并没有太多的描述，也引出了关于大模型推理技术的发展。

### **Reasoning技术：ReACT**

为了教会模型实用工具，一种方法是首先让模型具备推理的能力，从而能够模拟人使用工具的过程。应该说语言模型的训练方式和推理是不沾边的，但是语言模型的美妙之处就在于，当模型大小足够大的时候，它会诞生出很多出乎意料的能力，比方说推理能力。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f9802e42df4b47b38052f77048cf4bc6~tplv-k3u1fbpfcp-watermark.image?)


大语言模型的推理能力通过Chain-of-thought体现出来，但是这种推理能力需要显式的Prompt进行引导。根据引导方式的不同产生出各种不同的技术，其本质上是对不同思维方式的模拟，这里我们只介绍比较典型的ReACT技术。

ReACT用强化学习的方式建模推理的过程，agent认为是一个可以使用各种工具的智能体，environment为所有可用插件构成的工具箱集合，action为可以使用的插件功能集合。而控制策略为语言模型中学习到的知识。一个典型的推理的流程如下图所示：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8a73d01fcff34b82a86d03f2b71fb5c8~tplv-k3u1fbpfcp-watermark.image?)


ReACT推理流程可以分为Thought→Action→Observation→Thought 这样的循环，具体如何实现在本文的后续内容中会进行分析。

### **使用工具的语言模型：Toolformer**

与利用推理能力使用工具的思路不同，Toolformer是在训练语言模型过程中，使模型学习在适当位置调用相关API，并用API结果辅助后续的文本生成。在Toolformer训练过程中，数据是`Pittsburgh is also known as [QA(What …?→ Steel City)] the Steel City.`这种格式，如果是人去标注数据，首先需要找到API的放置位置，判断标准是API结果对后续文本生成有帮助，并且上文中有API需要的参数；然后是将API的标识、输入、输出以`[QA(What …?→ Steel City)]`这种形式插入到训练文本中。

注意，模型训练仍然采用典型的文字接龙方式，所以对原本语言模型的能力并没有损失。论文中提出一种利用LLM去自动标注这种数据的方式，和远程监督类似，步骤如下图：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/40bb5084a59a43d987bc9cd9d5ec1853~tplv-k3u1fbpfcp-watermark.image?)


### **工具提出：LangChain**

LangChain差不多是在2022年底提出的，那时候也是LLMs技术急剧发展的阶段。其核心是做一个基于LLMs的工具，基本上所有需要用LLMs实现的功能都可以在里面找到对应的工具。其中一个主要的能力，就是教会模型使用工具，并且接入方式和扩展性都非常好。除此之外还有很多好用的工具，比如：Prompt管理、Memory。名字中的Chain表示其核心设计思路是将不同的模块链接在一起。

详细的文档见[链接](https://github.com/hwchase17/langchain)。

<https://github.com/hwchase17/langchain>

LangChain中有很多有用的工具，包括各种搜索引擎Bing、Google、SerpAPI（google问答）、wiki等。还有一个更有趣的是Human as a tool插件，可以使语言模型必要时询问人类，从而模拟各种各样的功能。

在原理部分我们会介绍它的工作流程。

## ChatGPT plugin的原理

ChatGPT plugin是作为一个产品发布的，并且功能还没有完全开放，因此其实现原理细节也不是很清楚。但是由于LangChain中已经实现了类似的功能，并且2者的发布时间比较相近，所以有理由相信2者在原理上是有相似的。

下面我们分2部分，首先分析LangChain中使用工具的原理；第二部分通过比较2者的区别，得出一些关于ChatGPT plugin原理的猜想。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ece2e5b64a4f40f8a2e4d538f37cdbfe~tplv-k3u1fbpfcp-watermark.image?)



### LangChain的工作流程

首先让我们看一个例子：

```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

response = agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

代码里涉及到的关键概念：

-   llm: BaseLLM

    LangChain中对一系列开源模型接口的封装，其主要作用是统一不同的模型API，使接口更易使用，包括像提供缓存等一些基础功能。

-   tools: List[BaseTool]

    对工具的封装，LangChain中对工具的封装是比较简单的，因此保证了比较高的自由度，唯一的要求是输入输出必须是文字形式，`def run(self, tool_input: str) -> str`。

    自定义Tool只需要3个参数:

    -   name：工具的标识名称；
    -   description: 工具的自然语言描述；
    -   func: 功能执行函数，输入输出都为单个的文本。

-   agent: Agent

    内部使用了一个LLM决定使用什么工具，LangChain中agent的实现有2种，一种是ReACT类型，一种是self-Ask类型，因为后者只能使用qa类型的工具，如果任务涉及不同类型的工具，最好用ReACT类型。其中比较常用的是`zero-shot-react-description`，其中zero-shot表示推理引导Prompt里不包括示例，description表示LLM在决定调用什么工具的信息都来自于工具的description字段。

    注意，针对特定的任务可以设计针对性的few-shot提升agent的效果。

介绍完上面的概念，让我们看这个例子是怎么工作的。首先根据提供的工具，agent会生成引导Prompt，对于上面的例子，prompt是下面的样子：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a1234d5ea48c47838dd4dc8d8477ccca~tplv-k3u1fbpfcp-watermark.image?)

其中`{input}`为用户Query的占位符号，`{agent_scratchpad}`为模型生成填充的位置。下面说明一个循环Thought→Action→Observation→Thought的详细步骤：

-   生成Thought，对应的Prompt（只显示Begin!之后的部分）：

    > Question: Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?
    >
    > Thought:

-   LLM输出：

    > I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power. Action: Search Action Input: "Leo DiCaprio girlfriend"
    >
    > Observation:

    其中`Observation:` 为语言模型生成的终止符。

-   根据模型选择的Action，调用Search[ "Leo DiCaprio girlfriend"]得到结果：

    > Leonardo DiCaprio has split from girlfriend Camila Morrone. Getty. The Titanic actor hasn't been in a relationship with a woman over the age of ...

-   第二次生成Thought，对应的Prompt如下：

    > Question: Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?Thought: I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power. Action: Search Action Input: "Leo DiCaprio girlfriend" Observation: Leonardo DiCaprio has split from girlfriend Camila Morrone. Getty. The Titanic actor hasn't been in a relationship with a woman over the age of ... Thought:

-   继续这个循环直到输出最终结果，或者超过最大循环次数。

最后，完整的推理过程如下：

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/403cd9564d10443cb850f3ef086a307b~tplv-k3u1fbpfcp-watermark.image?)

### ChatGPT plugin的原理猜想（未完）

根据OpenAI官方的介绍，ChatGPT plugin在设计上要比LangChain精细的多，主要体现：

-   每个插件可以有多个API接口；
-   接口可以定义参数类型和格式；
-   描述的长度更大；

并且按照描述，对于自定义的新插件使用起来也是zero-shot的方式，所以其实现难度要更高。根据一些相关文献，可以猜想出以下可能的实现方式：

待补充…

## 讨论

### 应用场景

手机行业

机器人行业

### 对劳动力的影响

Zippia. "23+ Artificial Intelligence And Job Loss Statistics [2023]: How Job Automation Impacts the Workforce" [Zippia.com](http://zippia.com/). Feb. 7, 2023, <https://www.zippia.com/advice/ai-job-loss-statistics/>

这段内容摘抄自上面这篇博客，作者是一家求职网站的创始人。文中用到的并非严格的统计方法，因此数字有夸大的嫌疑，这里过滤掉一些数字表示的结论，总结出一些未来可能的趋势，供大家参考。

-   被AI技术淘汰掉旧的劳动力，不太可能找到更高薪的工作；
-   从长远来看，AI技术发展对全球经济是有促进作用的。但如果造成大规模的失业潮，就另当别论；
-   大多数公司会使用不同程度的AI技术来提升效率，也会是AI技术发展最直接的受益者；
-   当AI技术、机器人替代掉大多数的工作，很多人会没有工作，从而需要政府救济维持生活；
-   最有可能被取代的工作类型：客服、会计、前台接待、制造类、零售接待、数据分析 等；
-   最难被取代的工作类型：HR、作家、律师、管理者、科学家、人文类 等；

## 参考

1.  [Welcome to LangChain — 🦜🔗 LangChain 0.0.132](https://python.langchain.com/en/latest/)
1.  MRKL System [2205.00445.pdf (arxiv.org)](https://arxiv.org/pdf/2205.00445.pdf)
1.  REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS
1.  Toolformer：[*](https://arxiv.org/pdf/2302.04761.pdf)<https://arxiv.org/pdf/2302.04761.pdf*>
1.  ReACT：<https://arxiv.org/pdf/2210.03629.pdf>
1.  [YouTube@aiadvantage](https://www.youtube.com/@aiadvantage)