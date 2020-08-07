---
title: hexo的markdown图片文件路径转换
date: 2020/08/07 22:40:55
tags:
  - hexo
---

虽然hexo是通过书写markdown，然后再`hexo g`生成静态的网页文件，但是有个问题就是，图片放在什么地方？引申一下，其他的资源文件放在什么地方。

其实`hexo`官方也给出了自己的方案：[hexo资源文件夹](https://hexo.io/zh-cn/docs/asset-folders)，但是还需要手动写两条，有点麻烦。其实解决方法也很简单，装个hexo的插件就行了，详细的看[文档](https://github.com/cocowool/hexo-image-link)吧。

话说回来，hexo的文档资源管理有点怪怪的，为啥不能创建一个文件夹，里面放着资源文件以及markdown文件，而是将markdown文件和资源文件(创建一个资源文件夹)分开呢？

