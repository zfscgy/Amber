# What is Amber

Amber(**A** **m**achine learning system **b**ehaved in an **e**ncrypted manne**r**) is a library for privacy-preserving machine learning.

What is privacy-preserving machine learning? For example, suppose there are two companies, company A has some data $\mathbf X_1 \in \mathbb R^{n\times m}$, and Company B  ssome data $\mathbf X_2 \in \mathbb R^{n \times k}$ and $\mathbf Y \in \mathbb R^{n \times 1}$. And each row in $\mathbf X$ (either $\mathbf X_1$ or $\mathbf X_2$) represents some information about one customer. And elements in $\mathbf Y$ represents whether the customer are interested in some particular goods. Company B wants to train a model $M$ predict $\mathbf Y$ for new customers for a *recommendation task*, so it can uses its own customer information $\mathbf X_2$. However, for better prediction, it wants to use $\mathbf X_1$ which contains exact same users' extra information. But Company A refuses to directly give $\mathbf X_1$ to Company B. It worries about that although Company B promises to only use $\mathbf X_1$ for its current *recommendation task*, but maybe someday Company B will sell those $X_1$ to the others since Company B already had those data. So how can Company A use its data $\mathbf X_1$ to assist Company B's recommendation task while being sure that $\mathbf X_1$ will not be put in other use by Company B? So here comes the privacy-preserving machine learning.

In a short word, PPML(privacy-preserving machine learning) is to conduct machine learning in a secure manner that the data for training models will not be exposed.

# The idea behind Amber

Recently, many PPML  systems were created, e.g. Facebook's Crypten, Pysyft, Tensorflow's tf-encrypted, FATE, and CryptFlow. However, those PPML systems are stuck with one specific platform, heavy to install and hard to use. For example, Crypten is the lightest system among those all. However, it is hard to use it for PPML in production since its *semi-honest third-party* is simulated locally, which means it is unsafe. Pysyft is for the horizontal-federated tasks. tf-encrypted is hard to use since it's based on the tf 1.x's functions and there are plenty of bugs. FATE and CryptFlow are very heavy with their docker images of several GBs size. And they are also hard for beginners to use.

Different from all those PPML systems, Amber is a light-weighted  system that has no forcible requirements except for *Numpy*. Basic computations such as addition, multiplication and machine learning models such as LR, DNN are separated. As long as one protocol supports basic tensor operations, it can be used. And even those MPC protocols can use different backends for local computation. Amber contains the following layers:

* Layer 0: Backends for basic local computations. For example, *Numpy*.
* Layer 1: Players. It specifies a protocol by assigning different players in the protocol different tasks. Players can use different backends to accelerate their computations. An instance of Player should support common computations such as *add*, *mul*, *matmul*.
* Layer2: Operators. An operator is a computation which are differentiable. That is: the operator's derivative is still an operator. It is the basic element of the computation graph.
* Layer3: Graphs. A graph consists of operators, tensors which are parameters. And it takes some tensors as input, then outputs some tensors as output. Graphs are differentiable, so for one graph, we can get the gradient of it. Then we can update the parameters in that graph, in other words, training it.

