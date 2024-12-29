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
