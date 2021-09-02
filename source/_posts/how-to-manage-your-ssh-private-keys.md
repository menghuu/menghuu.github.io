---
title: how-to-manage-your-ssh-private-keys
date: 2021-09-02 18:47:29
tags:
    - ssh
    - ssh-add
    - ssh-keygen
    - openssh
    - win10
    - git-for-windows
    - git
    - keepassxc
    - keepass
    - KeeAgent
---

# 如何正确的管理你的ssh key

这是困扰我很久的问题，具体包括

1. 是否应该一台client一个单独private key？
3. 是否应该一台server一个单独public key？(一个很奇怪的问题)
5. 我是否应该在某个安全的地方备份我的private key？
6. 我是否应该在某个安全的地方备份我的public key？
7. 经常看到 ssh-agent 以及 ssh-add 命令，我从来都没用过，怎么回事？
8. 我是否应该使用`~/.ssh/config`来配置访问用的私钥

这几个问题其实交叉在一块，总是令我感觉难受，这里尽量的去回答这些问题。

----------------------------


首先需要说明私钥与公钥的关系：ssh使用的是[双钥加密，公钥和私钥是一一对应的关系，有一把公钥就必然有一把与之对应的、独一无二的私钥，反之亦成立](http://www.ruanyifeng.com/blog/2006/12/notes_on_cryptography.html)，所以说，以下很多的时候私钥也是公钥，公钥也是私钥，密钥也是指他们，但是更加侧重于一对的味道

对于第一个问题，以及第二个问题，其实就是在问，我在生成公钥私钥的时候是只考虑client，还是只考虑server，还是两者都考虑

那思考一下，这三种思考方式的结果会有什么优势和劣势。以下假设，每台client都要拥有每台server的访问权限前提（这个是一个比较苛刻的条件，毕竟有时候不会让公司设备client访问我的私人项目server，反过来，私人设备client访问公司server还是挺常见，但也不是每台私人client都需要访问每台公司server）。

| 情形（m个client，n个server，一般m << n）                     | 最多多少对密钥对 | 每台client保存的密钥数量，衡量方便程度 | 一台client失信时，server端以及其他的client端需要操作次数，可以看作某种风险和便捷性综合指标 | 优势                                                         | 劣势                                                         | 使用场景                                     |
| ------------------------------------------------------------ | ---------------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| 1. 相同的client，对于不同的server，使用一对密钥；不同的client，使用不同的密钥对 | m                | 1                                      | n+1+n（n个server去掉失信client的公钥，然后，失信client重新生成密钥对上传到n个server中） | client设备丢失或者泄露私钥，直接在server去除这个公钥即可；client设备不需要考虑众多的server情况 |                                                              | 如果手动管理私钥，推荐使用                   |
| 2. 不同的client，对于相同的server，使用一对密钥；不同的server，使用不同的密钥对 | n                | n                                      | n+m+m\*n（n个server首先去掉失信client的公钥，然后m个client更换私钥，然后同步到n个server中） | 可以直接解除某个**人**对某个server的全部访问权限             | server太多的话，client会保存很多的密钥；解除危险client对server的访问权的时候，其他的client设备也需要更换密钥对 |                                              |
| 3. 不同的client，对于不同的server，使用不同的密钥            | m \* n           | n                                      | n+1+n（n个server首先去除失信client的公钥，然后失信client重新生成密钥，然后上传到n个server） | 可以直接解除某个**设备**对某个server的访问权限               | server太多的话，client会保存**更**多的密钥；                 |                                              |
| 4. 不同的client，对于不同的server，使用一对密钥              | 1                | 1                                      | n+m+m\*n（n个server首先去除失信公钥，然后m个client更换私钥，然后同步到n个server中） | 十分方便，client端只会保存一个私钥                           | 一旦丢失一个，全部完蛋                                       | 推荐使用keepass之类的第三方工具来管理ssh key |



首先，根据上述的比较，似乎可以直接舍弃情况2，因为它在丢失client密钥或者密钥不被信任时，处理的操作也过于繁琐，而且，在不同的client之间拷贝密钥手动处理有点麻烦，如果你使用的是某种无加密的同步方式，则不安全，使用了某种加密的同步方式（比如keepass的keeagent插件），那么你为何不只是用一个被保护的很好的key呢？也就是为什么不使用情况4呢？

其次，由于绝大多数情况下，我们的密钥都是被（不可恢复地）丢弃了，而不是被不被信任了，情况1需要进行n次操作，情况3需要n次操作，情况4不需要额外操作。这里的操作是说把server中的公钥删除。如果是不被信任了，则，需要操作的步骤数量是列在表格中了。似乎在密钥被丢失的情形下，情况4有点胜出，但是出于还有可能出现client不被信任的情况，则还是得看看情况1与情况3的具体比较。

情况3在server比较少的时候，手动一个一个生成密钥对，然后添加到server中，看似不是那么麻烦，但是，如果server真的数量很少，为什么还要单独设置密钥呢？如果server数量很多的话，麻不麻烦不多说，真的需要删除server中的public key的时候，是否还能记起是哪些server添加了这台client的哪个公钥？尤其是当你还删除了公钥的文件的情况下，所以还是推荐情况1。

所以可以有个结论了：情况1与情况4是最好的，他们的共同点是，不区分server，即不是很在乎server的情况，优先考虑client的情况。那么给出一个可实践的结论

1. 如果只是你这个人在使用，并且，你使用的ssh客户端是你信任的，那么你可以只使用一对共同的密钥
2. 是否区分私人server和公司server，不是那么重要，但是因为密钥的备注问题，可以适当的分开，毕竟两个主要密钥不是太多
3. 如果是你不放心的ssh client，比如某些破解的ssh client，则给其设置单独的密钥
4. 如果是某些自动化机器人、ci、cd流程，则给其设置单独的密钥
5. 推荐使用一个很强的保护私钥的同步工具，推荐使用keepass+keeagent或者keepassxc
6. 如果不想依赖于第三方工具，则，自己在手动拷贝这个唯一的私钥时，尽快的`ssh-add`，然后立即删除，不要留在网盘或者u盘中
7. 不必过于担心安全，虽然我看到很多的从内存中提取ssh-agent保存的私钥，但是与其担心这个，不如一切小心，不要运行来历不明的软件，尤其是那些需要管理员权限的不明来历的软件，尽量少使用破解软件（好像我自己都没能遵守）
8. 这些推荐使用方式，似乎绝大多数人都是这么干的，还是应了那句话：如无必要，勿增实体。



以上给出了关于第一个第二个问题的答案。那么，对于第三个问题，其实也解决了，就是：你需要同步你的私钥，而且是加密的同步，或者手动复制之后，立刻`ssh-add`，然后立即删除。



对于第四个是否应该备份公钥，答案是，不必要，保存私钥即可生成公钥，具体的命令为：`ssh-keygen -y -f ~/.ssh/id_rsa`，如果你想更改密钥的备注，可以使用`ssh-keygen -c -f ~/.ssh/private_key`来重新生成新的备注，备注不会影响私钥本身的验证性（至少github不会影响可验证性），会在私钥文件的最后几位进行改变



问题5，经常看到 ssh-agent 以及 ssh-add 命令，我从来都没用过，怎么回事？

简单来说，ssh-agent是加密你的私钥，在你使用ssh命令时自动解密成私钥后走ssh的流程，不需要你手动`ssh -i /path/to/private/key git@github.com:username/git-repo.git`

ssh-add 是用于添加你的私钥到加密的ssh-agent的“数据库”中，建议你add之后，在你明白到底发生了什么事情之后，看懂这篇文章之后，立刻删除这个私钥



问题6，我是否应该使用`~/.ssh/config`来配置访问用的私钥

我不建议，但是建议你使用config来配置其他的选项。



TODO:

1. WSL中的ssh也能够使用windows自带的ssh-agent，或者反过来也行
   - [A Better Windows 10+WSL SSH Experience - Shea Polansky](https://polansky.co/blog/a-better-windows-wsl-openssh-experience/)
   - [rupor-github/wsl-ssh-agent: Helper to interface with Windows ssh-agent.exe service from Windows Subsystem for Linux (WSL)](https://github.com/rupor-github/wsl-ssh-agent)
   - [rupor-github/win-gpg-agent: Windows helpers for GnuPG tools suite - OpenSSH, WSL 1, WSL2, Cygwin, MSYS2, Git4Windows, Putty...](https://github.com/rupor-github/win-gpg-agent)
   - [benpye/wsl-ssh-pageant: A Pageant -> TCP bridge for use with WSL, allowing for Pageant to be used as an ssh-ageant within the WSL environment. (github.com)](https://github.com/benpye/wsl-ssh-pageant)
2. 好像还有其他的更加丰富的加密验证方式
   1. 




------------------------
参考链接:
1. [Should I generate new SSH private keys for each machine I work from?](https://security.stackexchange.com/questions/102416/should-i-generate-new-ssh-private-keys-for-each-machine-i-work-from)
2. [key management - What's the common pragmatic strategy for managing key pairs? - Information Security Stack Exchange](https://security.stackexchange.com/questions/10963/whats-the-common-pragmatic-strategy-for-managing-key-pairs?rq=1)
3. [privacy - Best Practice: ”separate ssh-key per host and user“ vs. ”one ssh-key for all hosts“ - Information Security Stack Exchange](https://security.stackexchange.com/questions/40050/best-practice-separate-ssh-key-per-host-and-user-vs-one-ssh-key-for-all-hos)
4. [如何从Windows 10 ssh-agent中提取SSH私钥 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37390404)

