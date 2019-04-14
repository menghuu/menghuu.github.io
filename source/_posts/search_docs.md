---
title: search docs
tags:
    - zeal
    - dash
    - document
    - tool
    - api
    - search
---

我从小记性就差，每次要写点代码，都会打开许多的网页去搜索相关的api是怎么用的，但是实际上这种事情怎么可能没人发现，怎么会没有相关的工具去解决这个问题呢？

在mac下，有[Dash](https://kapeli.com/dash)，在windows下有类似的工具[zeal](https://zealdocs.org)，后者不仅支持windows，也有linux和mac的客户端。并且zeal使用的document的格式也与Dash通用(实际上，zeal不维护任何的document，只是在用Dash维护的document，感谢Dash的作者，这位作者对zeal也有贡献)。除了Dash类的，还有[devdocs](devdocs.io)这一类的网页版的document搜索工具。

说来惭愧，如果我真的使用mac的话，估计也会使用zeal或者devdocs这一类免费的工具，而不会去使用Dash这一类收费的软件。当然我也正在使用windows，目前来说，没啥纠结的。

devdocs这个工具实际上也不错，只是没有我想要的pytorch文档，其实我之前为devdocs贡献过pytorch的文档，但是我一直不太会ruby，也对js不慎了解，其实官方对于贡献的要求比较高，我就直接发了个帖子，把我做的pytorch半成品发了上去，后来也没去管了。

我最近才发现，原来Dash也有第三方的pytorch文档，虽然那个样式真是惨不忍睹，但是zeal这个工具貌似不直接支持Dash的[第三方文档贡献](https://github.com/Kapeli/Dash-User-Contributions)。我在[这里](https://github.com/zealdocs/zeal/issues/170#issuecomment-477661338)找到有解决办法，简单来说，就是直接从[这里](http://zealusercontributions.herokuapp.com/)搜索相关的文档，然后将xml链接feed到zeal中。

似乎这篇文章没啥技术含量