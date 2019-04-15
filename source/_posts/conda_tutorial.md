---
title: conda/anaconda安利+速记手册
date: 2019-04-11 14:36:58
tags: 
    - docker
    - python
    - anaconda
---


`anaconda`和`python`的关系和`ubuntu`和`linux kernel`的关系十分像，准确来说，`anaconda`是`python`的一个面向数据科学的`发行版`。而`conda`就像`apt-get/apt`之于`ubuntu/debian`，是`pip`功能的超集。

-------

问：为什么使用`anaconda`？ 

答：因为包含一些常用的科学计算包，你不需要安装额外的这些东西，想想，你如果每次都需要安装`numpy/pandas/matplotlib/jupyter/qtconsole/ipython`，是不是要崩溃？当然，如果你不喜欢这么多包，想自己按照需要安装，可以使用`miniconda`这个简化版的`anaconda`。

--------

问：为什么使用`conda`？ 

答：除了能像`pip`一样安装包之外，还具备虚拟环境的功能。如果你不使用`anaconda`或者`minconda`环境，除非自己安装`conda`，在python中，只能通过`pyenv`或者`pipenv`或者`virtualenv`之类的工具创建虚拟环境。对于安装包，我还想说，`conda`不仅仅能安装python的包，也能安装r的包(它自我标榜是一个系统级的包管理工具，但是谁会用conda管理系统级的软件？)，一般情况下，如果使用pip有依赖问题，使用`conda`都可以轻松解决。很多的工具也都全力支持anaconda环境，对pip的支持不够好，比如`spyder`，顺便安利下`spyder`，我个人觉着`spyder`/`vscode`/`pycharm`/`jupyter notebook`是不同的东西，但都是好工具。

--------

问：我为什么要使用虚拟环境？

答：只要是下过别人的开源代码的都知道(尤其是使用tensorflow写的代码)，解决依赖是一件非常头痛的事情，每个程序都依赖于不同的包(这还不算大事)，甚至是同一个包的不同的版本(严重问题)，如果只有一个环境，您如何解决问题？即使不使用conda创建虚拟环境，我还是建议每一个复杂的项目都建立自己的环境。

--------
<!--more-->

上述就是使用`conda`的原因。

那么，不使用`conda`的原因呢？

- 兼容性。一般开源的代码中直接给出`requirements.txt`文件，里面是其依赖，为了通用性，一般都是使用`pip`导出的，因为`conda`下载的包的地址和`pip`的不同(不太严谨的说：`conda`默认是从anaconda.org搜索，`pip`是从pypi.python.org)，所以`conda`能够安装的python包和`pip`能安装的不太一样，甚至名字都不一样。我认为这是conda最大的问题。
- 代码/工具洁癖。事实上，anaconda是一个商业python发行版，它是有pro版本的，即使`conda`这个工具是开源的。

---------

你是否想继续下去？觉着太麻烦？不想装anaconda？也不想装miniconda？

那么，你可以使用`pip install conda`来安装`conda`，使用`conda`来安装某些难装的包，但请注意，这样的操作我是没有实际使用过的，但是这是可行的，我不知道这样是不是被推荐的，反正我不是很推荐。conda和pip安装的包不能相互操作

## 安装anaconda/minconda/conda并配置

建议从国内的镜像中下载并配置国内的源，这里推荐两个`http://mirrors.ustc.edu.cn/help/anaconda.html`和`https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/`网址，里面有anaconda/miniconda的下载地址以及相应的国内源配置情况。除此之外，还可以选择aliyun或者上海交通大学的软件园或者你们自己学校的或者其他的什么软件源。顺便说下，清华和ustc都有pypi的源，也建议修改。

## 使用conda

- 默认是在base环境下，使用`conda activate my_env`来激活自己的环境
- `conda create --name my_env pytorch` 创建自己的环境并安装pytorch
- `conda create --name myclone --clone myenv` clone一份myenv 到新环境myclone中
- `conda deactivate`退出当前的环境
- `conda install tensorflow==2.0-alpha`安装相应的包
- `conda env remove --name torch_env`
- 未完待续


## 推荐如何使用

- 安装anaconda
- 在基础的环境下安装`tensorflow`和`pytorch`，顺便说下，在所有的操作系统下，需要自行安装了相应的英伟达驱动，但是不需要自己安装cuda和cudnn，conda都会自己安装这些依赖到特定的虚拟环境下，不用担心会污染全局环境
- 使用conda克隆虚拟环境`conda create --name torch_env --clone base`，这样能减少向源下载包的情况
- 激活新的环境`conda activate torch_env`
- `conda install new_package`或者`pip install -r requirements.txt`

注意：clone的话，你的新环境可能会占用更多的空间，如果某个环境不用了，请及时删除`conda env remove --name torch_env`
