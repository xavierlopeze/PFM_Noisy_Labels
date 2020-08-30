## Deep Learning with Noisy Labels
This repository is the result of a final Master's Thesis of the Master's Degree in Master's Degree in Fundamental Principles of Data Sciencee supervised by Petia Radeva, PhD and Javier Ródenas. 

In this thesis we explore techniques to make Deep Learning robust on noisy datasets. With noisy datasets we mean datasets that contain samples wrongly labeled. We will work on  supervised classification problems of mutually exclusive classes (i.e. without overlap). We tackle computer vision problems, although those techniques are model-agnostic and can be implmeneted on a general classification task.
We particulary focus on MLNT and VAEs.

Our results suggest that on datasets with and without noisy labels MLNT is able to consistently outperform a conventional cross-entropy learning approach on different kinds of noise.


### Noisy label training methods:
* [MLNT](https://github.com/LiJunnan1992/MLNT)

### Datsets:
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Food-101](https://www.kaggle.com/dansbecker/food-101/home)

### Delivered on 02-09-2020 by:
* Jordi Ventura
* Xavier López
