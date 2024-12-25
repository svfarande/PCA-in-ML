# Week 1
Let's say 
* **Dataset** : $ D = \{x_1, x_2, .... , x_N\} $  then 
## Mean
* The mean of a data set describes the average data point.
* The mean does not have to be a typical data point and it also does not need to be part of the data set itself 
* **Mean** : $$ Mean[D] = \frac{1}{N}\sum_{n=1}^{N}x_n $$
* Let's say $\bar{x}_{n-1}$ is mean of dataset $D_{n-1}$ with $n-1$ data points. Now suppose we collect another data point which is denoted by $x_*$ then the new mean $\bar{x}_n$ with another data point $x_*$ is given by : 
$$ \bar{x}_n = \bar{x}_{n-1} + \frac{1}{n}(x_* - \bar{x}_{n-1} ) $$
* If we multiply each sample in dataset $ D $ by factor $f$ then the mean of the new dataset will be $f$ times the mean of D. 
## Variance
* The variance is used to characterise the variability or spread of data points in a dataset.
* **Variance** : $$ Var[D] = \frac{1}{N}\sum_{n=1}^{N}(x_n - \mu)^2 $$
where $\mu$ is $ Mean[D] $
* Variance cannot be negative.
* Square root of variance is **Standard Deviation** : $$ \sigma = \sqrt{Var[D]} $$  
* If we multiply each sample in dataset $ D $ by factor $f$ then the variance of the new dataset will be $f$ times the variance of D. 
* Let's say $\bar{x}_{n-1}$ is mean and $\sigma^2_{n-1}$ is variance of dataset $D_{n-1}$ with $n-1$ data points. Now suppose we collect another data point which is denoted by $x_*$ then the new variance $\sigma^2_{n}$ with another data point $x_*$ and new mean $\bar{x}_n$ is given by : 
$$ \sigma^2_{n} =  \frac{n-1}{n} \sigma^2_{n-1} + \frac{1}{n}(x_* - \bar{x}_{n-1} ) (x_* - \bar{x}_{n}) $$
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
* The covariance matrix is always a symmetric positive definite matrix, with the variances on the diagonal and the cross covariance or covariances on the off diagonals.
* If the covariance between $x$ & $y$ is positive, then on average the $y$ value increases if we increase $x$.  
And if the covariance between $x$ and $y$ is negative, then the $y$ value decreases if we increase $x$ on average.  
If the covariance between $x$ and $y$ is zero, $x$ and $y$ have nothing to do with each other. They are uncorrelated. 
---
* Let's say D is M x N dimensional data consisting of N vectors each of length N given by :
$ D = \{X_1, X_2, .... , X_M\} $
then $$ CoVar[D] = \frac{1}{M}\sum_{i=1}^{N}(x_i - \mu)^T(x_i - \mu) $$
where $\mu$ is mean of the dataset and $T$ is transpose of a vector
* If we multiply dataset by factor $f$ then covariance and variance both are multipled by $f^2$