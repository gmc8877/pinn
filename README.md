# use pinn to solve poisoon function

solve a 2D poission function like this:
  ```math
    \begin{align}\
  -\Delta u(\boldsymbol{x}) & = f(x), & & \boldsymbol{x} \in \Omega, \\
  u(\boldsymbol{x}) & = 0, & & \boldsymbol{x} \in \partial \Omega .
  \end{align}
  ```
```math
\begin{equation}
f(\boldsymbol{x})=-32 \pi^{2} \prod_{i=1}^{2} \sin \left(4 \pi x_{i}\right)
\end{equation}
```
and a 10D poission function:
```math
\begin{equation}
f(\boldsymbol{x})=-160 \pi^{2} \prod_{i=1}^{10} \sin \left(4 \pi x_{i}\right)
\end{equation}
```
