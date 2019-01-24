---
title: find的使用
tags:
  - find
  - linux
date: 2019-01-07 01:37:15
---


> find [-H] [-L] [-P] [-D debugopts] [-Olevel] [starting-point...] [expression]

前面的几项是什么意思，可以直接man查看，大概是处理链接这种情况，`starting-point` 是开始搜索的目录,就是字面意思。 重点是`expression`是啥

<!--more-->

# expression

`expression`分为以下几类  
- `Tests` return true或者return false，比如`-empty`测试是否是空文件  
- `Actions`, 有副作用，比如`-print`  
- `Global options` 影响整个命令行的动作，比如`-depth`这个参数是用来指定find使用深度优先搜索  
- `Positional options` 值影响那些跟在他之后的操作。Positional options永远都是返回true，比如`-regextype`  
- `Operators`, 比如`-o`、`-a`，默认是`-a`   

如果整个表达式没有除了`-prune`或者`-print`之外的动作，那么`-print`将会应用在所有那些在整个表达式结果是true的文件上

## Position Options

- -daystart  
- -follow  
- -regextype type  这个有关使用的regextype，会影响`-regex`对filename的匹配，**但是不会影响-name之类的正则表达式** 可以使用-regextype help 来查看具体的种类，[gnu document](https://www.gnu.org/software/gnulib/manual/html_node/Regular-expression-syntaxes.html#Regular-expression-syntaxes)有具体的正则表达式的具体用法  
- warn, -nowarn  

## Global options

- -d, -depth  
- help, --help  
- -ignore_readdir_race
- -maxdepth levels
- mindepth levels
- mount
- noignore_readdir_race  
- noleaf 这个关乎性能，可能在windows下使用  
- version, --version
- xdev

## Tests

不想抄了，简单说说，类似`-amin +n`之类，测试文件的属性的，常见的属性有
- -emtpy  
- name  永远不能有`/`符号, 这个的语法是[shell pattern  ](https://www.gnu.org/software/findutils/manual/html_node/find_html/Shell-Pattern-Matching.html)或者[这里](http://wiki.bash-hackers.org/syntax/pattern)
- path  
- type c, c的种类有 `b`(block buffered special), `c`(character unbuffered special), `d`(directory), 'p'(pipe), `f`(regular file), `l`(symbolic link), `s`(socket), `D`(door)

## Actions

- delete  
- exec command;  
- exec command {} +  
- execdir command ;
- execdir command {} +   # 这些我没用到所以没有仔细看，应该还是有点东西的
- -fls file  # 和ls类似，只是ls到一个文件中
- -fprint file  
- -fprint0 file  
- -fprintf file format  
- -ls  
- -ok command ;  
- -okdir command ;  
- -print True;  
- -print0  
- -printf format  

## operators

- ( expr )  
- ! expr  
- -not expr  
- expr1 expr2
- expr1 -a expr2  
- expr1 -and expr2  
- expr2 -o expr2  
- expr1 -or expr2  
- expr1 , expr2  # expr1的结果直接被忽略  

## 遇到奇怪的文件名时该怎么办

- -print0, -fprint0 直接输出那些奇奇怪怪的文件名  
- -ls, fls 会进行escaped  
- -printf, fprintf 主要是为了format  
- -print, -fprint  
- -ok / -okdir 别用了  


----

本文是我在配置我的dotfiles的时候，为了通用一些，写了一些shell脚本来进行安装的时候，需要用到`find`命令，我以前一直觉着需要看很多的东西，但是在我浏览了manpage之后，竟然发现出乎意料的短。manpage真好用。