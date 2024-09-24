# landmark
Shift and Mode Detection

* A new approach, a max landmark-sliced kernel, is presented to compute Wasserstein distance. 
* The proposed method can be computed efficiently for the case of two samples, unlike max-sliced (kernel).
* Landmark method is computationally suitable for high-dimensional data.
* We showed our proposed method could detect shifts locally, where the baseline method MMD explains the data globally. 





# Abstract
Recently, slicing techniques have been proposed to provide computational and statistical advantages for the Wasserstein distance for high-dimensional spaces. In this work, we presented a computationally simple approach to perform slicing of the kernel Wasserstein distance and apply it as a readily  interpretable two-sample test. The proposed landmark-based slicing chooses a single point from the two samples to be the single support vector to represent the witness function. We run experiments using modified MNIST and CIFAR10 dataset and compare our method with maximum mean discrepancy (MMD). We investigate various shift scenarios and the effect of the choice of learning representations. The results show that our proposed methods perform better than MMD on these synthetic simulations of covariate shift.

https://www.eecis.udel.edu/~ajbrock/other/karahan_poster_distshift2021 <br/>
https://openreview.net/pdf?id=Wu5hMMQ76OE



# To run landmark with failing loudly:
 We test our method using code from https://github.com/steverab/failing-loudly. The followings are required to run this repo. 

https://github.com/steverab/failing-loudly <br />
keras: https://github.com/keras-team/keras <br />
tensorflow: https://github.com/tensorflow/tensorflow <br />
pytorch: https://github.com/pytorch/pytorch <br />
sklearn: https://github.com/scikit-learn/scikit-learn <br />
matplotlib: https://github.com/matplotlib/matplotlib <br />
torch-two-sample: https://github.com/josipd/torch-two-sample <br />
keras-resnet: https://github.com/broadinstitute/keras-resnet <br />
POT: https://pythonot.github.io/ <br />


The shift_tester.py file updated so that our MLW method can be compared to the MMD. 
Notice the original code provided by https://github.com/steverab/failing-loudly calculates the median kernel size for MMD incorrectly. We updated that too. 

Make sure the failing loudly and landmark files are in the same directory. 



Run all experiments using:
`run_pipeline.py`

python3.8 pipeline_landmark.py DATASET SHIFT DIMENSIONALITY method <br />
Example: python3.8 pipeline_landmark.py mnist adversarial_shift multiv LMSKW

