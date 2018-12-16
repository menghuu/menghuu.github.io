---
title: compile_tensorflow_from_src
tags: linux tensorflow compile
date: 2018-12-17 05:02:56
---

# 从源码编译tensorflow

1. 本文编译不考虑独立显卡部分的相关库的安装
2. 基于ubuntu18.04，使用的是gcc5
3. 基于anaconda的python3(具体采用的是python3.5)
4. 如果不是由特殊的要求，强烈建议使用tensorflow官方的二进制包，比如pip
5. 实际上如果只是想将tensorflow中的模型和参数应用到实际受限的环境，需要编译成lib文件，或许有更好的方式，而不是使用这个方法。
6. 写的是整个步骤，所以太啰嗦了，如果你已经安装好bazel，那么直接拉到本文最后查看具体的安装步骤吧。

<!--more-->

----

参考资料  
> tensorflow的官方从源码编译tensorflow的步骤流程 https://www.tensorflow.org/install/source  
> 安装balze的官方资料： https://docs.bazel.build/versions/master/install-ubuntu.html  

----

我这里首先装了bazel，我是安装在自己的家目录中的。具体的  
> sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python  
> 从 https://github.com/bazelbuild/bazel/releases 中下载适合系统的\*install\*.sh  
> chmod +x bazel-<version\>-installer-linux-x86_64.sh  
> ./bazel-<version\>-installer-linux-x86_64.sh --user #安装bazel  
> export PATH="$PATH:$HOME/bin"

这个类似于cmake，tensorflow在linux和mac中是使用bazle进行构建，在windows中使用cmake

----

根据tensorflow官方的资料，还需要安装如下的包  
> sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel  

这些包的作用是
> numpy：这是 TensorFlow 需要的数值处理软件包。  
> dev：用于向 Python 添加扩展程序。  
> pip：用于安装和管理某些 Python 软件包。  
> wheel：用于管理 wheel (.whl) 格式的 Python 压缩包。

~~但是我是在anaconda3的环境下安装的，所以这些默认都是安装好的。~~我的anaconda中的python版本是3.7的，这个支持不是特别好，所以最终采用的是python3.5

从描述上来看，这些包的作用主要是为了增加tensorflow的python api的扩展能力，没有他们(除了numpy之外)似乎对于tensorflow的cpp lib没有什么影响。

----

根据官方的给的教程，我使用`git clone https://github.com/tensorflow/tensorflow`下载了源码，并使用`git checkout r1.13`来选择r1.13的tensorflow的版本,你可以根据自己的需要选择相应的版本。

如果采用的是r1.0,然后根据按照官方教程中`./configure`来配置，出现[这个问题](https://github.com/tensorflow/tensorflow/issues/16654)，好像是因为判断bazel的版本号时候出现了bug，需要打上两个补丁。好像只要是r1.5以前的都有这个问题，我本来使用的是r1.0，结果打上这两个补丁也不能用。所以最终还是使用了1.13的版本。r1.13版本不需要打这两个补丁。

> git cherry-pick 3f57956725b553d196974c9ad31badeb3eabf8bb  
> git cherry-pick 6fcfab770c2672e2250e0f5686b9545d99eb7b2b

实际上，这个r1.13依旧不能很好的编译成功，主要是因为我是在python3.7中编译的，在python3.7中async以及await成了关键词，但是代码中还是没有修改好。或许这个[commit](https://github.com/tensorflow/tensorflow/pull/21202/files)可以解决?（忘记了能不能行）

----

然后就是`./configure`了,我这里没有指定cuda，如果有需要，请一定要选择。

现在如果没有错误，就是说tensorflow配置好了，这个时候使用下面命令使用bazel build
```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```
因为我是使用cpu的版本，所以，如果是使用gpu版本的，使用
```bash
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
```

一段非常非常长的等待，抱歉，最终使用python3.7+tensorflow-r1.13还是没有成功。

----

最后使用的是py35 + keras_preprocessing成功编译成功tensorflow-r1.13。具体的步骤
```bash
conda create -n py35 python=3.5 pip numpy wheel  
conda activate py35  
pip install keras_preprocessing #很重要，否则还是不能成功编译  
git clone https://github.com/tensorflow/tensorflow  
cd tensorflow  
git checkout r1.13  
./configure  
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package  
```

我在服务器上使用40个逻辑核心的计算机也跑了很长时间，内存占用也十分多，所以最好能够一次就能编译好，否则，等待时间实在是太长了。