- [Week 1](#week-1)
  - [Mean](#mean)
  - [Variance](#variance)
  - [CoVariance](#covariance)
- [Week 2](#week-2)
  - [Dot product](#dot-product)
    - [Length of $X$](#length-of-x)
    - [Eucledian Distance between $X$ \& $Y$](#eucledian-distance-between-x--y)
    - [Angle $\\alpha$ between $X$ \& $Y$ in $radians$](#angle-alpha-between-x--y-in-radians)
  - [Inner Product](#inner-product)
    - [Bilinear](#bilinear)
    - [Positive Definite](#positive-definite)
    - [Symmetric](#symmetric)
  - [Inner Product : Length of vectors](#inner-product--length-of-vectors)
    - [Properties](#properties)
  - [Inner Product : Distances between vectors](#inner-product--distances-between-vectors)
  - [Inner Product : Angles and Orthogonality](#inner-product--angles-and-orthogonality)
- [Week 3](#week-3)
  - [Projection onto 1D subspaces](#projection-onto-1d-subspaces)
  - [Projection onto higher dimensional subspaces](#projection-onto-higher-dimensional-subspaces)
- [Week 4](#week-4)
  - [Problem Setting and PCA Objective](#problem-setting-and-pca-objective)
  - [Reformulation of the PCA Objective](#reformulation-of-the-pca-objective)
  - [Finding the optimal basis vectors that span the principal subspace](#finding-the-optimal-basis-vectors-that-span-the-principal-subspace)
  - [Steps of PCA](#steps-of-pca)


# Week 1
Let's say 
* **Dataset** : $ D = \{x_1, x_2, .... , x_N\} $  then 

## Mean
* The mean of a data set describes the average data point.
* The mean does not have to be a typical data point and it also does not need to be part of the data set itself 
* **Mean** : $$ Mean[D] = \frac{1}{N}\sum_{n=1}^{N}x_n $$
* Let's say $\bar{x}_{n-1}$ is mean of dataset $D_{n-1}$ with $n-1$ data points. Now suppose we collect another data point which is denoted by $x_*$ then the new mean $\bar{x}_n$ with another data point $x_*$ is given by : 
$$ \bar{x}_n = \bar{x}_{n-1} + \frac{1}{n}(x_* - \bar{x}_{n-1} ) $$
* If we shift each sample in dataset $ D $ by factor $a$ then the mean of the new dataset will be $a$ times the mean of D: $$Mean[D+a] = a+Mean[D]$$ 
* If we multiply or scale each sample in dataset $ D $ by factor $f$ then the mean of the new dataset will be $f$ times the mean of D: $$Mean[D*f] = f*Mean[D]$$ 
* Thus :  $$Mean[D*f + a] = f*Mean[D] + a$$ where $f$ is multiply / scale & $a$ is shift
* For higher dimension : $$Mean[D.F + A] = F . Mean[D] + A$$ where $F$ is multiply / scale vector & $A$ is shift vector.

## Variance
* The variance is used to characterise the variability or spread of data points in a dataset.
* **Variance** : $$ Var[D] = \frac{1}{N}\sum_{n=1}^{N}(x_n - \mu)^2 $$
where $\mu$ is $ Mean[D] $
* Variance cannot be negative.
* If we multiply or scale each sample in dataset $ D $ by factor $f$ then the variance of the new dataset will be $f^2$ times the variance of $D$ :  $$Var[D*f] = f^2*Var[D]$$
* If we shift each sample in dataset $ D $ by factor $a$ then the variance of the new dataset won't change: $$Var[D+a] = Var[D]$$
* Thus :  $$var[D*f + a] = f^2*Var[D]$$ where $f$ is multiply / scale & $a$ is shift.
* Let's say $\bar{x}_{n-1}$ is mean and $\sigma^2_{n-1}$ is variance of dataset $D_{n-1}$ with $n-1$ data points. Now suppose we collect another data point which is denoted by $x_*$ then the new variance $\sigma^2_{n}$ with another data point $x_*$ and new mean $\bar{x}_n$ is given by : 
$$ \sigma^2_{n} =  \frac{n-1}{n} \sigma^2_{n-1} + \frac{1}{n}(x_* - \bar{x}_{n-1} ) (x_* - \bar{x}_{n}) $$
* Lets day $D_1$ & $D_2$ are two datasets having $Var[D_1]$ < $Var[D_2]$ then we can conclude that spread of $D_2$ is more than $D_1$.
* Square root of variance is **Standard Deviation** : $$ \sigma = \sqrt{Var[D]} $$ 
---
*Note* : The standard deviation is expressed in the same units as the mean value whereas the variance unfortunately is expressed in squared units.

## CoVariance 
*Variance in High Dimensional Dataset*  
Lets say $x_i$ are points in $X$ direction & $y_i$ are points in $Y$ direction then : 
* Covariance between any point $x$ & $y$ is defined as : $$CoVar[x,y]= (x-\mu_x)(y-\mu_y)$$ where  
$\mu_x$ is $Mean[x_i]$ in $X$ direction   
$\mu_y$ is $Mean[y_i]$ in $Y$ direction
* Therefore for 2D data in $X$ & $Y$ direction with any point $x$ & $y$ we can have $Var[x]$, $Var[y]$, $CoVar[x,y]$ & $CoVar[y,x]$.  
We can represent this in 2 x 2 matrix as below : 
$$
\left(\begin{array}{cc} 
Var[x] & CoVar[x,y]\\
CoVar[y,x] & Var[y]
\end{array}\right)
$$
* If data is N dimensional then the covariance matrix is N x N matrix
* The covariance matrix is always a **symmetric positive definite matrix**, with the variances on the diagonal and the cross covariance or covariances on the off diagonals.
* If the covariance between $x$ & $y$ is positive, then on average the $y$ value increases if we increase $x$.  
And if the covariance between $x$ and $y$ is negative, then the $y$ value decreases if we increase $x$ on average.  
If the covariance between $x$ and $y$ is zero, $x$ and $y$ have nothing to do with each other. They are uncorrelated. 
* The covariance of a matrix won't change if we add a vector to data.
---
* Let's say D is M x N dimensional data consisting of M vectors each of length N given by :
$ D = \{X_1, X_2, .... , X_M\} $
then $$ CoVar[D] = \frac{1}{M}\sum_{i=1}^{N}(x_i - \mu)^T(x_i - \mu) $$
where $\mu$ is mean of the dataset and $T$ is transpose of a vector
* If we multiply dataset by factor $f$ then covariance of new dataset is multipled by $f^2$ : $$CoVar[D*f] = f^2*CoVar[D]$$
* For higher dimension : $$CoVar[D.F + A] = F.CoVar[D].F^T$$ where $F$ is multiply / scale vector, $A$ is shift vector and $F^T$ is transpose of $F$.

# Week 2
## Dot product
In order to measure angles, lengths, and distances, we need to equip the vector space with an inner product which allows us to talk about geometric properties in a vector space. An example of an inner product that we may know already is the dot product between two vectors, $X$ and $Y$ : 
$$X^T.Y = \sum_{i=1}^{N}x_i*y_i$$

### Length of $X$
$$||X|| = \sqrt{X^T.X} = \sqrt{\sum_{i=1}^{N}x^2_i}$$

### Eucledian Distance between $X$ & $Y$
$$d(X,Y) = ||X-Y|| = \sqrt{(X-Y)^T.(X-Y)}$$

### Angle $\alpha$ between $X$ & $Y$ in $radians$
$$cos \alpha = \frac{X^T.Y}{||X|| * ||Y||} $$
Thus,
$$\alpha = cos^{-1}(\frac{X^T.Y}{||X|| * ||Y||})$$

## Inner Product
* Inner product between $X$ & $Y$ vector is given by : $$< X, Y > = X^T . A . Y$$ where A should be  :
    * **symmetric** [ $A^T=A$ ] and  
    * **positive definite** [ All eigenvalues should be positive i.e $determinant(\lambda I - A) = 0$ where $\lambda \ge 1$ ].
* Dot Product is special case of Inner Product where $A$ is $I$ the Identity Matrix.  
--- 
All 3 (**symmetric**, **positive definite** & **bilinear**) are necessary for a function to qualify as inner product :

### Bilinear 
Let's say for vector $X$, $Y$ & $Z$ and $\lambda$ is the Real number then bilinearity is : $$ <\lambda X + Z , Y> = \lambda <X , Y> + <Z , Y> $$
$$ <X, \lambda Y + Z> = \lambda <X , Y> + <X, Z> $$

### Positive Definite
A inner product with itself is always >= 0 : $$ <X, X> \ge 0 $$ & $$ <X, X> = 0 \Longleftrightarrow X = 0 $$

### Symmetric
Inner product of $X$ , $Y$ & $Y$, $X$ is same.
$$ < X, Y > = < Y, X > $$

## Inner Product : Length of vectors
* The length of $X$ is also called the **norm of $X$** given by : $$|| X || = \sqrt {<X, X>}$$
* For a inner product $< X, X > = X^T . A . X$
    * If $A = I$ (i.e. *dot product*) then : $$|| X || = \sqrt{X^T.X} = \sqrt{\sum_{i=1}^{N}x^2_i}$$
    * But if $A \ne I $ then :  $$ || X || = \sqrt {<X, X>} =  \sqrt { X^T . A . X}$$
### Properties 
* $$||\lambda X|| = |\lambda| * ||X||$$
* **Triangle inequality** : $$ || X + Y || \le ||X|| + ||Y||$$
* **Cauchy-Schwart inequality** : $$|<X,Y>| \le ||X|| * ||Y||$$

## Inner Product : Distances between vectors
* Distance $d$ between any 2 vector $X$ & $Y$ is given by difference of these vectors : $$d(X,Y) = ||X-Y|| = \sqrt{<X-Y , X-Y)>} = \sqrt{(X-Y)^T. A. (X-Y)}$$ this is **un-squared or not squared distance**.
* **Squared Distance** : $$d(X,Y)^2 = ||X-Y||^2 = <X-Y , X-Y)> = (X-Y)^T. A. (X-Y)$$
* If we use dot product then distance is called as an Eucledian Distance.
* Depending on the choice of our inner product, we will get different answers of what the distance between $X$ and $Y$ actually is.
* Distance using dot product will be equal to distance using inner product only if $A = I$
## Inner Product : Angles and Orthogonality
* Angle $\omega$ between vector $X$ & $Y$ is given by : $$cos \omega = \frac{<X, Y>}{||X|| * ||Y||} $$
Thus,
$$\omega = cos^{-1}(\frac{<X, Y>}{||X|| * ||Y||})$$
* Two vectors, $X$ and $Y$, where $X$ and $Y$ are non-zero vectors, are **orthogonal** (i.e. $\omega$ is $\frac{\pi}{2} rad$ or $90^\degree$) if and only if their inner product is zero. 
    * This also means that orthogonality is defined with respect to inner product. And vectors that are orthogonal with respect to one inner product do not have to be orthogonal with respect to another inner product.
* From a geometric point of view, we can think of two orthogonal vectors as two vectors that are most dissimilar and have nothing in common besides the origin. We can also find a basis of a vector space such that the basis vectors are orthogonal to each other. That means, we get the inner product between $b_i$ and $b_j$ is zero if $i$ is not the same index as $j$. And we can also use the inner product to normalise these basis vectors. That means, we can make sure that every $b_i$ has length one. Then we call this an **orthonormal basis**. That is :   
$<b_i, b_j> = 0$ ....... if $i \ne j$ & $||b_i|| = 1$

# Week 3
## Projection onto 1D subspaces
Let's say we have : 
1. vector $X$ in 2D
2. 1D subspace $U$
3. with basis vector $b$ in it *i.e.* all vectors in $U$ can be represented by $\lambda b$ for sure.  

We are interested in finding a orthogonal projection of $X$ onto $U$. That means the difference vector of $X$ and its projection is orthogonal to $U$. Let's denote this projection by $\pi_U(X)$. This projection has 2 important properties :  
1. $\pi_U(X) = \lambda b$           .   .   .  ......... where $\pi_U(X) \in U => \exist$  $\lambda \in R$
2. $<b, \pi_U(X) - X > = 0$ *[orthogonality]*

Now $\lambda$ can be given as : $$\lambda = \frac{<b,X>}{||b||^2}$$
Thus, $$\pi_U(X) = \lambda b = \frac{<b,X>}{||b||^2} b = \frac{bb^T}{||b||^2}X$$
Where - $$ \frac{bb^T}{||b||^2}$$ is **projection matrix** which is square and symmetric matrix. 
We can also write it as $$\pi_U(X) = \lambda b = \frac{<X,b>}{||b||^2} b = \frac{X^Tb}{||b||^2}b $$

## Projection onto higher dimensional subspaces

Let's say we have : 
1. vector $X$ in 3D
2. 2D subspace $U$
3. with basis vector $b_1$ and $b_2$ in it *i.e.* all vectors in $U$ can be represented by $\lambda_1 b_1$ + $\lambda_2 b_2$ for sure.

Now we are interested in finding a orthogonal projection of $X$ onto $U$. That means the difference vector of $X$ and its projection is orthogonal to $U$. Let's denote this projection by $\pi_U(X)$. This projection has 2 important properties :  
1. $\pi_U(X) = \lambda_1 b_1$ + $\lambda_2 b_2$           .   .   .  ......... where $\pi_U(X) \in U => \exist$  $\lambda_1 , \lambda_2 \in R^2$
2. $<b_1, \pi_U(X) - X > = 0$ & $<b_2, \pi_U(X) - X > = 0$ *[orthogonality]*

Now, let's formulate our intuition for the general case, where $X$ is a $D$-dimensional vector, and we are going to locate an $M$-dimensional subspace $U$.  
Thus, 
$$
\lambda = 
\left(\begin{array}{cc} 
\lambda_1\\
\lambda_2\\
\..\\
\lambda_M\\
\end{array}\right)_{M\times1}

B = 
\left(\begin{array}{cc} 
\ b_1 &|& b_2 &|& ...&|&  b_M\\
\end{array}\right)_{D\times M}
$$
Thus our two important properties now will be :
1. $\pi_U(X) = \sum_{i=1}^{M}\lambda_i b_i = B\lambda$ .   .   .  ......... where $\pi_U(X) \in U => \exist$  $\lambda_i \in R^2$
2. $<b_i, \pi_U(X) - X > = 0$ , $i=1, 2. . .M$ => *[orthogonality]*

Now $\lambda$ can be given as : $$\lambda = (B^TB)^{-1}B^TX$$
Thus, $$\pi_U(X) = B\lambda = B(B^TB)^{-1}B^TX$$
Where - $$ B(B^TB)^{-1}B^T$$ is **projection matrix**.   

In special case of **Ortho Normal Basis** $B^TB = I$ Then $$\pi_U(X) = BB^TX$$ where $$BB^T$$ is **projection matrix**.

# Week 4

## Problem Setting and PCA Objective
Lets say there is **centered** (i.e. $Mean(X)=0$) $X$ data set in $R^D$.   
$X = \{x_1, ... x_N \} , x_i \in R^D $  
Our objective is to find a low dimensional representation of the data that is as similar to $X$ as possible.

We have below three properties :

1. Every vector in $R^D$ can be represented as a linear combination of the basis vectors $b_i$ : $$\begin{equation} x_n = \sum_{i=1}^D \beta_{in}b_i \end{equation}$$
where $b_i$ are orthonormal basis of $R^D$

2. If we assume we use dot product over inner product and $b_1$ to $b_D$ are orthonormal basis then we can write : $$\beta_{in} = x_n^Tb_i$$ which means we can interpret $\beta_{in}$ to be the orthogonal projection of $x_n$ onto the one dimensional subspace spanned by the $i^{th}$ basis vector. 

3. If we have an **orthonormal basis** $b_1$ to $b_M$ of $R^D$ i.e. $B = (b_1 , ... , b_M)$ and we define $B$ to be the matrix that consists of these orthonormal basis vectors. Then the projection of $X$ onto the subspace, we can write as : $$\tilde{X} = BB^TX$$
where $B^TX$ are co-ordinates or code in basis $B$

Splitting sum in property point 1 into two sum such that one is in $M$ dimensional subspace and the other one is in $D$ dimensional subspace which is orthogonal complement to $M$ dimensional subspace : 
$$\begin{equation} \tilde{x}_n = \sum_{i=1}^M \beta_{in}b_i +  \sum_{i=M+1}^D \beta_{in}b_i \end{equation}$$
In PCA we ignore the 2nd term and call the 1st term from $(b_1 , ... , b_M)$ as principal subspace given by : $$ \begin{equation} \tilde{x}_n = \sum_{i=1}^M \beta_{in}b_i \end{equation} $$   

Assuming we have data $X$, we want to find parameters $\beta_{in}$ and orthonormal basis vectors $b_i$, such at the average squared re-construction error $J$ is minimised (loss function). And we can write the average squared re-construction error as follows : $$ \begin{equation} J = \frac{1}{N} \sum_{n=1}^N||x_n - \tilde{x_n}||^2 \end{equation} $$

Our approach is to compute the partial derivatives of $J$ with respect to the parameters. The parameters are the $\beta_{in}$ and the $b_i$. We set the partial derivatives of $J$ with respect to these parameters to zero and solve for the optimal parameters. The parameters only enter this loss function through $\tilde{x}_n$ tilde. This means that in order to get our partial derivatives, we need to apply the chain rule. So, we can write : $$\frac{\partial J}{\partial \{\beta_{in}b_i\}} = \frac{\partial J}{\partial \tilde{x}_n}. \frac{\partial \tilde{x}_n}{\partial \{\beta_{in}b_i\}}$$
The first part will be : $$\begin{equation} \frac{\partial J}{\partial \tilde{x}_n} = - \frac{2}{N}(x_n - \tilde{x}_n)^T\end{equation}$$

The second part will be :  $$\frac{\partial \tilde{x}_n}{\partial \{\beta_{in}b_i\}} = b_i $$

Thus, $$\begin{equation} \frac{\partial J}{\partial \{\beta_{in}b_i\}} = - \frac{2}{N}(x_n - \tilde{x}_n)^T b_i \end{equation} $$ 

Now substituting $\tilde{x}_n$ from equation (3) in equation (6) and solving it further we get : $$\frac{\partial J}{\partial \{\beta_{in}b_i\}} = - \frac{2}{N}(x_n^Tb_i - \beta_{in}) = 0$$

As we will need to solve for parameter $\beta_{in}$ we will need to make above equation equal to 0. The equation will be 0 only if : $$\begin{equation}\beta_{in} = x_n^Tb_i\end{equation}$$

## Reformulation of the PCA Objective

 Substituting equation (7) in equation (3) and solving it further we get : $$\begin{equation} \tilde{x}_n = \sum_{i=1}^M b_ib_i^T x_n \end{equation}$$
where the projection matrix is : $$\sum_{i=1}^M b_ib_i^T$$

Similarly we can substitute equation (7) in equation (1) and split it like we did it for $\tilde{x}_n$ in equation (2) we get : $$ \begin{equation} {x}_n = (\sum_{i=1}^M b_ib_i^T)x_n + (\sum_{i=M+1}^D b_ib_i^T)x_n \end{equation}$$

Now subtracting $\tilde{x}_n$ equation (8) from $x_n$ equation (9) we get : 
$$\begin{equation}x_n - \tilde{x}_n = (\sum_{i=M+1}^D b_ib_i^T)x_n = \sum_{i=M+1}^D (b_i^Tx_n)b_i \end{equation}$$

Now reformulating our loss function $J$ i.e. equation (4), lets substitute equation (10) in loss function and solve it further we get : 
$$J = \sum_{i=M+1}^D b_i^T (\frac{1}{N} \sum_{n=1}^Nx_nx_n^T) b_i$$

We can see the middle part is **data covariance matrix** $S$ : 
$$S = \frac{1}{N} \sum_{n=1}^Nx_nx_n^T$$


Thus : $$\begin{equation} J = \sum_{i=M+1}^D b_i^T S b_i \end {equation}$$
where $S$ is symmetric, positive definite, & the vector-matrix product is linear and hence $b_i^T S b_i$ is inner product.

Now re-arranging the terms using $trace$ operator we can re-write $J$ as : $$J = trace((\sum_{i=M+1}^D b_ib_i^T ) S )$$

Hence the projection matrix is given by : $$\sum_{i=M+1}^D b_ib_i^T$$ This projection matrix takes our data covariance matrix $S$ and project it onto the orthogonal compliment of the principal subspace. That means, we can reformulate the loss function as the variance of the data projected onto the subspace that we ignore. Therefore, minimising this loss is equivalent to minimising the variance of the data that lies in the subspace that is a orthogonal to the principal subspace.

## Finding the optimal basis vectors that span the principal subspace

Let us start with an example to determine the $b_i$ basis vectors and let's start in two dimensions, where we wish to find a one dimensional subspace such at the variance of the data when projected onto that subspace is minimized. Lets say we have two basis vectors $b_1$ and $b_2$, $b_1$ will be spanning the principle subspace and $b_2$ its orthogonal complement, that means the subspace that we will ignore. We also have the constraint that $b_1$ and $b_2$ are orthonormal.  
From equation (11) : $$ \begin{equation} J = b_2^TSb_2 \end{equation} $$ and we need to obtimize this using Lagrange Multiplier ($\lambda$): $$L = b_2^TSb_2 + \lambda (1 - b_2^Tb_2)$$
Computing gradiants for $L$ wrt $b_2$ & $\lambda$ and setting them to 0 : 
$$\begin{equation} \frac{\partial L}{\partial \lambda} = 1 - b_2^Tb_2 = 0   \Longleftrightarrow b_2^Tb_2 = 1 \end{equation} $$ 
$$\begin{equation} \frac{\partial L}{\partial b_2} = 2b_2^TS - 2\lambda b_2^T = 0  \Longleftrightarrow Sb_2 = \lambda b_2 \end{equation}$$

Here we end up with an eigenvalue problem where - $b_2$ is eigenvector and $\lambda$ is eigenvalue

Going back to loss function equation (12) and substituting value of $Sb_2$ from equation (14) in it and further substituting from equation (13) as well we get : $$J = b_2^Tb_2\lambda = \lambda$$

The loss can be minimized by choosing $\lambda$ to be the smallest eigenvalue of the covariance matrix.  The smallest value is associated with the least loss of information. Therefore, the average squared reconstruction error is minimized if $\lambda$ is the smallest eigenvalue of the data covariance matrix. And that means we need to choose $b_2$ as the corresponding eigenvector, and that one will span the subspace that we will ignore. $b_1$ which spans the principle subspace is then the eigenvector that belongs to the largest eigenvalue of the data covariance matrix.

## Steps of PCA
[Understanding Principle Component Analysis(PCA) step by step.](https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9)

We're given a two dimensional data set, and we want to use PCA to project it onto a one dimensional subspace. 

1. Subtract the mean
2. Divide by the standard deviation
3. We compute the data covariance matrix and its eigenvalues and corresponding eigenvectors.
4. The eigenvectors are scaled by the magnitude of the corresponding eigenvalue. The longer vector spans the principal subspace.
5. We can project any data point $x_*$ onto the principal subspace.

**Projection** -
$$\tilde{x}_* = \pi(x_*) = BB^Tx_*$$
where $B$ is the matrix that contains the eigenvectors that belong to the largest eigenvalues as columns. And $B^Tx_*$ are the coordinates of the projection with respect to the basis of the principal subspace. 

## PCA in high dimensions

