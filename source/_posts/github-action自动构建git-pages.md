---
title: github action自动构建git pages
date: 2021-09-02 22:46:38
tags:
    - github
    - page
    - action
    - blog
    - ci
---
github action自动构建git pages

我之前是使用travis的，但是不知为何现在travis-ci好久不能成功了，我现在连无法登录管理我之前的自动化部署我的博客了，所以，采用了github自带的action这个ci工具来自动将我hexo分支的md文件生成静态文件，并push到master分支中，由此来部署我的博客。

------------------------

<!-- more-->

具体的yml文件，可以见https://github.com/menghuu/menghuu.github.io/blob/hexo/.github/workflows/deploy.yml

```
name: Deploy CI

on:
  push:
    branches: [ hexo ]
  pull_request:
    branches: [ hexo ]

jobs:
  build:

    runs-on: ubuntu-latest
    env:
      TZ: 'Asia/Shanghai'
      GH_REF: github.com/menghuu/menghuu.github.io.git  #设置GH_REF
      NODE_VERSION: 14
    steps:
    - uses: actions/checkout@v2
    - name: setup Node.js
      uses: actions/setup-node@v2
      with:
        cache: 'npm'
        
    - name: Cache node modules
      uses: actions/cache@v2
      with:
        path: ./node_modules
        key: ${{ runner.os }}-build-${{ env.cache-name }}-${{ hashFiles('**/package-lock.json') }}
        restore-keys: |
          ${{ runner.os }}-build-${{ env.cache-name }}-
          ${{ runner.os }}-build-
          ${{ runner.os }}-
          
    - name: install packages
      run: |
        npm install -g hexo
        npm install
    - name: generate static pages
      run: |
        hexo cl
        hexo g
    - name: git commit page
      run: |
        cd ./public
        git init
        git config user.name "meng.hu"
        git config user.email "mail@meng.hu"
        git add .
        git commit -m "Site deployed by Github Action"
        git push --force --quiet "https://menghuu:${{ secrets.GITHUB_TOKEN  }}@${{ env.GH_REF }}" master:master
```

其实没有什么太困难的，只是有些文章会让你创建person access key使得你的ci工具可以push文件，或者有的会设置一个单独的ssh key，目的也是为了让你的ci工具能够push。其实github本身为action创建了一个access token，那就是`GITHUB_TOKEN`。[工作流程中的身份验证 - GitHub Docs](https://docs.github.com/cn/actions/reference/authentication-in-a-workflow)

所以在这个里面，你不需要设置任何别的东西，只需要注意一下你文章的MD文件所在的分支（在我的例子里面是hexo），以及最终会将生成的静态html去push到哪个分支中（在我的例子中，是master）。

除此之外，还是用了cache这个action，用于缓存nodejs的安装的packages。这个对于最终的CI跑完的速度还是有影响的，能省个十秒钟吧。

最后别忘了领个徽章哦。[添加工作流程状态徽章 - GitHub Docs](https://docs.github.com/cn/actions/managing-workflow-runs/adding-a-workflow-status-badge)



又水了一篇

-----------------

参考链接(有关于cache的)

- [Super fast npm install on Github Actions (voorhoede.nl)](https://www.voorhoede.nl/en/blog/super-fast-npm-install-on-github-actions/)

- [缓存依赖项以加快工作流程 - GitHub Docs](https://docs.github.com/cn/actions/guides/caching-dependencies-to-speed-up-workflows)
- [actions/cache: Cache dependencies and build outputs in GitHub Actions](https://github.com/actions/cache/)