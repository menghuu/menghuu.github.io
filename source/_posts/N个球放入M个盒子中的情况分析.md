---
title: N个球放入M个盒子中的情况分析
date: 2019-04-15 14:36:58
tags: 
  - 算法
  - 数据结构
---

对于情况分析，主要参考：

- [参考1](https://blog.csdn.net/u011244839/article/details/53443505)
- [参考2](https://blog.csdn.net/Jaster_wisdom/article/details/78506831)
- [排列组合](https://blog.csdn.net/l_0000/article/details/82560166)

<!--more-->

本文只是对参考链接的简单“复制”，最多会有比较详细的解释，不会有什么新的东西


>  该类问题涉及到三个因素，分别是球是否有区别、盒子是否有区别、盒子是否可以为空。所以大概可以将该问题分为以下八种情况：

>1.将n个无区别的球放入m个无标志的盒中，没有一个盒子为空，有多少种情况？  
>2.将n个无区别的球放入m个无标志的盒中，盒内数目不限制，有多少种情况？  

>3.将n个无区别的球放入m个有标志的盒中，没有一个盒子为空，有多少种情况？  
>4.将n个无区别的球放入m个有标志的盒中，盒内数目无限制，有多少种情况？  

>5.将n个有区别的球放入m个无标志的盒中，没有一个盒子为空，有多少种情况？  
>6.将n个有区别的球放入m个无标志的盒中，盒内数目不限制，有多少种情况？  

>7.将n个有区别的球放入m个有标志的盒中，没有一个盒子为空，有多少种情况？  
>8.将n个有区别的球放入m个有标志的盒中，盒内数目不限制，有多少种情况？  

## 将n个无区别的球放入m个无标志的盒中，没有一个盒子为空，有多少种情况？

### 解释

这里肯定要假设球的个数n要大于盒子的个数m，否则的话，必然有盒子是空的。

记录此时的情况个数为$P_{m}(n)$。由于此时任何一个盒子都不能是空的，所以必然有m个球放在了这m个盒子中，并且由于是无区别/无标志，所以这只有一种情况，也就是说$P_m(m)==1$。剩下的球怎么分等下再说，这个前提得有。

这里给出结论：

$P_m(n) = P_1(n-m) + P_2(n-m) + ... + P_{n-m}(n-m)$

这里给出第一项的解释：在这m个盒子中任何`1`个盒子中要装上这`n-m`个球。

除了$P_m(m)==1$之外还有$P_{m-1}(m)==1$和$P_1(1)==1$，具体不在解释。

### 代码实现

```cpp
#include <iostream>

/**
 * @brief 计算n个无区别的球放入m个无区别的盒子中有几种方法(不允许有空盒子)
 * 
 * @param m M个盒子
 * @param n N个球
 * @return int 次数
 */
int p(int m, int n)
{
    if (m > n)
    {
        return -1;
    }
    if (m == 1 || m == n || m == n - 1)

    {
        return 1;
    }
    int count = 0;
    for (int i = 1; i <= n - m; i++)
    {
        count += p(i, n - m);
    }
    return count;
}

int main()
{
    int N, M;

    std::cout << "input the number of boxes(M): ";
    std::cin >> M;
    std::cout << "input the number of balls(N): ";
    std::cin >> N;
    std::cout << "count is: " << p(M, N) << std::endl;
}
```


## 将n个无区别的球放入m个无标志的盒中，盒内数目不限制，有多少种情况？  

### 解释

貌似不是特别好想，直接给出结论，$P_{m}(n+m)$

### 代码实现，略

## 将n个无区别的球放入m个有标志的盒中，没有一个盒子为空，有多少种情况？  

$C_{n-1}^{m-1}$

```cpp
#include <iostream>

// 主要是如何计算排列组合

/**
 * @brief 计算
 * 
 * @param m 
 * @param n 
 * @return int 
 */
int n_arragement(int n, int start_n, int start_arragement)
{
    // 不进行某些判断了
    int result = start_n > 1 ? start_arragement : 1;
    int start_i = start_n > 1 ? start_n + 1 : 1;
    for (; start_i <= n; start_i++)
    {
        result *= start_i;
    }
    return result;
}

/**
 * @brief 计算n个无区别的球放入m个有区别的盒子中有几种方法(不允许有空盒子)
 * 
 * @param m M个盒子
 * @param n N个球
 * @return int 次数
 */
int p(int m, int n)
{
        // 不进行某些判断了
    /**
     * $C_{n-1}^{m-1} = \frac{(n-1)!}{(n-m)! * (m-1)!}$
     * 
     */
    int part1, part2, part3 = 1; // (n-1)!  larger smaller
    if(n-m < m-1){
        part3 = n_arragement(n-m, 1, 1);
        part2 = n_arragement(m-1, n-m, part3);
        part1 = n_arragement(n-1, m-1, part2);
    } else {
        part3 = n_arragement(m-1, 1, 1);
        part2 = n_arragement(n-m, m-1, part3);
        part1 = n_arragement(n-1, n-m, part2);
    }

    return part1 / (part2 * part3);
}

int main()
{
    int N, M;

    // std::cout << n_permutation(5, 3, 6) << std::endl;

    std::cout << "input the number of boxes(M): ";
    std::cin >> M;
    std::cout << "input the number of balls(N): ";
    std::cin >> N;
    std::cout << "count is: " << p(M, N) << std::endl;
}
```

## 将n个无区别的球放入m个有标志的盒中，盒内数目无限制，有多少种情况？  

$C_{m+n-1}^{m-1}$


## 将n个有区别的球放入m个无标志的盒中，没有一个盒子为空，有多少种情况？

略， $S(N, M)$ –第二类斯特林数

## 将n个有区别的球放入m个无标志的盒中，盒内数目不限制，有多少种情况？  

略，$S(N, 1) + S(N, 2) + S(N, 3) + … + S(N, M)$

## 将n个有区别的球放入m个有标志的盒中，没有一个盒子为空，有多少种情况？

略，$M! * S(N, M)$

## 将n个有区别的球放入m个有标志的盒中，盒内数目不限制，有多少种情况？ 

$m^n$