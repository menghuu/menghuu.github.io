---
title: 测试从joplin中自动拷贝文章到git page中
date: 15/09/2021 22:35
tags:
	- git
	- joplin
	- blog
	
---


原理：从joplin自己启动的server中，获取指定tag名称的文档，以及相关的resources(一般都是图片)，复制到指定的git本地仓库地址，然后走`git add && git commit && git push` 这一套流程，说起来也没啥特殊的。

对了，还有点替换md文件中的resource的链接的行为


<!-- more-->

这里不会做什么比较内容是否更新的操作，你需要自己确认自己的某篇文档是否应该被更新，如果不想更新，直接将配置中的tag在joplin中取消掉。

一旦添加过了，就不能修改文档名称！！否则会出现两篇文档

如果你想要使用github action自动构建git pages，你可以参看我另外一篇[文章](http://meng.hu/2021/09/02/github-action%E8%87%AA%E5%8A%A8%E6%9E%84%E5%BB%BAgit-pages/)

讲真，无聊的脚本，如果你想用的话，可以看[这里]()

以下是测试拷贝附件

![=========302afb7a980406e26a6ee184ba9435fd.png](测试从joplin中自动拷贝文章到gitpage中/302afb7a980406e26a6ee184ba9435fd.png)
