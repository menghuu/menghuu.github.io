---
title: conda + pipenv 配置深度学习开发环境
tags: 
    - conda
    - pipenv
    - python
    - 虚拟环境
date: 2019-07-29 00:00:00
---

- 前提: 安装的 cuda 和 cudnn 版本和驱动是匹配的, 不匹配的话, 绝对不可能解决, 涉及到 linux kernel 处的驱动问题, 这不是普通权限用户能解决的
- `conda create -n cuda8_cudnn5 cudatoolkit==8.0.0 cudnn==5.1.0`
- `pipenv install —python==3.6 tensorflow-gpu==1.0.0`
- `echo 'export LD_LIBRARY_PATH=~/anaconda3/envs/cuda8_cudnn5/lib:$LD_LIBRARY_PATH'  >> .envrc`
- `source ./.envrc` 或使用 direnv 自动 source 和 de-source, 否则的话, 进去需要手动 source, 不需要这个动态链接的时候, 自行修改`LD_LIBRARY_PATH` 环境变量

原理: 其实就是动态链接的时候, 可以指定链接的 so 文件的路径, 使用`LD_LIBRARY_PATH`来指定. 依次类推,别的库文件也可以使用 conda 来管理,比如 opencv 之类的.

注意: 创建 conda 环境的时候, 不要安装 python, 主要是因为安装 python会安装某些额外的 lib 文件, 这样的话, 有时候你使用 bash 的时候也会动态链接到这些 lib 文件上, 容易出现问题. 

## 为什么这么干?

- docker 中比较容易使用 pip 来安装依赖, 容易部署
- pip 是个更加通用的工具
- pipenv 能导出 100% 能被pip 识别的 requirements.txt, 但是 conda 导出的依赖有时候是 conda 上特有的包
- pipenv 是个更加好用和先进的工具
- conda 上有很多编译好的库, 无需我们手动管理多种二进制库

