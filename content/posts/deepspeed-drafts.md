+++
title = 'Deepspeed原理（手写笔记）'
date = 2023-07-05T23:42:36+08:00
draft = false
busuanzi = true
+++

## 前言
介绍了一下DeepSpeed的架构，以及部分重点内容的原理。  
其实是看DeepSpeed源码时候随便写的一段笔记，没时间整理并且写的很潦草，所以不太想发，但是框架的代码读起来不容易，里面知识点确实花了一些时间才弄明白。  
另外，也看到DeepSpeed框架在工作中使用越来越多，所以发出来给想要了解DeepSpeed原理的人一个参考，欢迎批评指正，献丑了。
## 正文
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7f60ae205d0b4654baa9472925331336~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/57950a5a53dd4ea4bf7bb87d50b10926~tplv-k3u1fbpfcp-watermark.image?)