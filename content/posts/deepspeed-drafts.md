+++
title = 'Deepspeed原理（手写笔记）'
date = 2023-07-05T23:42:36+08:00
draft = false
+++

## 前言
介绍了一下DeepSpeed的架构，以及部分重点内容的原理。  
其实是看DeepSpeed源码时候随便写的一段笔记，没时间整理并且写的很潦草，所以不太想发，但是框架的代码读起来不容易，里面知识点确实花了一些时间才弄明白。  
另外，也看到DeepSpeed框架在工作中使用越来越多，所以发出来给想要了解DeepSpeed原理的人一个参考，欢迎批评指正，献丑了。
## 正文
![015c5fa81f2345a99f49b080be566eec](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/015c5fa81f2345a99f49b080be566eec-20240423141901724.jpeg)
![5920def6862e4711b9b611244ed157ec](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/5920def6862e4711b9b611244ed157ec-20240423141908989.jpeg)
![2b7b8c81485048f7a74f803bd0a700e5](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/2b7b8c81485048f7a74f803bd0a700e5.jpeg)
![ddeb1a8c2d7a4f72bc84cd6dcbd9745a](https://raw.githubusercontent.com/dawson-chen/picgo-repo/master/ddeb1a8c2d7a4f72bc84cd6dcbd9745a-20240423141936728.jpeg)
