---
title: python_logging_实践
tags: 
    - python
    - logging
    - 日志
---

# python logging 实践

实际上有很多的文章都写的很好，这里只记录自己的一些实践，此文一直在草稿箱中，如果您能看到这篇文章，那说明已经趋近我自己的想要表达的了。

对于logging这个模块的使用，需要知道四个概念

> Logger 记录器，暴露了应用程序代码能直接使用的接口。  
> Handler 处理器，将（记录器产生的）日志记录发送至合适的目的地。  
> Filter 过滤器，提供了更好的粒度控制，它可以决定输出哪些日志记录。  
> Formatter 格式化器，指明了最终输出中日志记录的布局。

Handler 是属于Logger的(使用`logger.addHandler`)，Filter可以属于Logger(`addFilter`)，也可以属于Handler(`addFilter`)，实际上，logger会先使用Filter先filter一遍，然后使用Handler再处理一遍(里面自带了filter)，Formatter设置在Handler中的(`setFormatter`)。所以Formatter对于每个handler中只有一个。
