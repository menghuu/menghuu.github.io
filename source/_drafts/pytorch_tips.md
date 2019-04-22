---
title: pytorch一些小的tips
tags: 
    - pytorch
    - tips
---

some tips of pytorch

<!--more-->

- `model.eval()` 只会影响到dropout，batchnorm之类的，不会对grad的记录有任何的影响
- `with torch.set_grad_enabled(False)`配合着上面的`eval`来对模型进行验证，否则可能会记录一些额外的grad信息(？真的影响这么大吗)
- 上述来自[Does model.eval() & with torch.set_grad_enabled(is_train) have the same effect for grad history?](https://discuss.pytorch.org/t/does-model-eval-with-torch-set-grad-enabled-is-train-have-the-same-effect-for-grad-history/17183)
- `.data`和`.detach`的区别可以看下面的例子，至少从下面的例子我没看出有什么区别，不过据[The difference between ‘.data’ and ‘.detach()’?](https://discuss.pytorch.org/t/the-difference-between-data-and-detach/30926)所说，`.data`只是为了兼容而存在的，不要使用这个了，只使用`detach`
> You should always use .detach() if you want to detach a tensor from the graph. The other option .data is for older versions of PyTorch, and it is likely that it will be removed from the future versions of PyTorch.  
```python
import torch
print(torch.__version__) # 1.0.1
# default
x = torch.tensor([1.0], requires_grad = True)
y = x**2
z = 2*y
w = z**3

p = z
q = torch.tensor(([2.0]), requires_grad=True)
pq = p*q
pq.backward(retain_graph=True)

w.backward()
print('x.grad: ',x.grad) # 56
print('z.grad: ',z.grad) # None
print('p.grad: ',p.grad) # None

########################
# using .detach
x = torch.tensor(([1.0]),requires_grad=True)
y = x**2
z = 2*y
w= z**3

# detach it, so the gradient w.r.t `p` does not effect `z`!
p = z.detach() ####
q = torch.tensor(([2.0]), requires_grad=True)
pq = p*q
pq.backward(retain_graph=True)

w.backward()
print('x.grad: ',x.grad) # 48
print('z.grad: ',z.grad) # None
print('p.grad: ',p.grad) # None

########################
# using .detach and requires_grad_()
x = torch.tensor(([1.0]),requires_grad=True)
y = x**2
z = 2*y
w= z**3

# detach it, so the gradient w.r.t `p` does not effect `z`!
p = z.detach() ####
p.requires_grad_()
q = torch.tensor(([2.0]), requires_grad=True)
pq = p*q
pq.backward(retain_graph=True)

w.backward()
print('x.grad: ',x.grad) # 48
print('z.grad: ',z.grad) # None
print('p.grad: ',p.grad) # 2

########################
# using .data
x = torch.tensor(([1.0]),requires_grad=True)
y = x**2
z = 2*y
w= z**3

p = z.data ####
q = torch.tensor(([2.0]), requires_grad=True)
pq = p*q
pq.backward(retain_graph=True)

w.backward()
print('x.grad: ',x.grad) # 48
print('z.grad: ',z.grad) # None
print('p.grad: ',p.grad) # None


#######################
# using .data and .requires_grad_()
x = torch.tensor(([1.0]),requires_grad=True)
y = x**2
z = 2*y
w= z**3

p = z.data  ####
p.requires_grad_() #####
q = torch.tensor(([2.0]), requires_grad=True)
pq = p*q
pq.backward(retain_graph=True)

w.backward()
print('x.grad: ',x.grad) # 48
print('z.grad: ',z.grad) # None
print('p.grad: ',p.grad) # 2,虽然p和z的值是共享的，但是传播的时候不会修改z的grad值，仔细想想好像也有道理
```
- [如何冻结模型](https://discuss.pytorch.org/t/the-difference-between-data-and-detach/30926)，简单来说，就是将模型参数先载入，然后将某些层的requires_grad设置成False或者在优化器里面设定需要优化的参数的时候，去掉那些不需要的参数
> Yes, it does work when you add the parameters with requires_grad=True to the optimizer then setting to False after. You can also find out yourself by commenting out

> optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

> In the snippet above, since the previous optimizer contains all parameters including the fc2 with the changed requires_grad flag.

> Note that the above snippet assumed a common “train => save => load => freeze parts” scenario.

- 正如keras是tensorflow的high level api，pytorch也有些高级api，但是实际上，pytorch已经是如此的方便了，它的高级api更多的是在训练、验证、早停、记录上。pytorch的github上有tnt和ignite两个实现，在我看来他们都是一个东西，不过它们各自有自己的设计目标，在我看来都不是特别成熟。就像tensorflow除了keras还有很多的high level api实现，pytorch还有 Poutyne这个高级api实现。这里简单的对比这三者的区别。值得注意的是，由于原生pytorch已经比较成熟，所以这几类高级api用的人不够多，所以开发的也不够成熟。

|          |  |                                       |                                                      |
| -------- | ---------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
|          | [ignite](https://github.com/pytorch/ignite/) | [tnt](https://github.com/pytorch/tnt/) | [poutyne](https://github.com/GRAAL-Research/poutyne) |
| 官方支持 | Y                                              | Y | N |
| [callback][^callback]是否跟随模型 | 是 | 是 | 否 |
| 官方测试例子 | [mnist](https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist.py) | [mnist](https://github.com/pytorch/tnt/blob/master/example/mnist.py) | [mnist](https://github.com/GRAAL-Research/poutyne/blob/master/examples/mnist.ipynb) |
| 如何定义callback | 1. 通过装饰器 2. 通过add_event_handler | 通过engine.hook[when]=do_something_fn来添加 | 通过fit的时候传入callbacklist |
| 定义callback优势 | 装饰器真的很好用 | 没啥，中规中矩 | 和keras比较像，用类来描述callback，封装比较好 |
| 定义callback劣势 | 真的写callback(handler)的话，真的不太好看，尤其是那些横跨好几个stage的handlers。[ignite对于handler的讨论](https://github.com/pytorch/ignite/issues/37) | 没有ignite的装饰器那么好写，不过这个也不是啥大问题。和ignite一样，写横跨好几个stage的hook，真的不好看。 | callback略微舒服点，但是没有ignite的装饰器那么好写，不过这个也不是啥大问题 |
| callback定义灵活度 | 中 | 低 | 高 |
| other comparations todos |  |  |  |
| 推荐指数(满分5星) | 3 | 3 | 3 |

[^callback]: ignite中callback被称作handlers，tnt中被称之为hooks，poutyne中被称之为callback和keras一致，事实上，poutyne更像keras。这里的是否跟随模型，是指跟随此高级api框架的模型，对于ignite和tnt来说是engine(为了不和pytorch的模型混淆，后面称之为engine)，对poutyne来说被称作framework的model(我已经给官方提了issue来说明这个词容易混淆)。对于此问题，到底是跟随engine好，还是不跟随engine好呢？个人以为需要结合编写handler的容易成都来判断，至少我感觉ignite实在是太难写handler了，尤其是跨好几个state的handler，需要重写蹩脚的attach来维持某些变量状态，否则的话，就需要像和tnt一样一次书写到处拷贝。从这个角度来看，还是poutyne更好一点。

