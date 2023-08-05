# Literature Review
# Table of Contents
1. [Performance comparison of deep learning frameworks](#performance-comparison-of-deep-learning-frameworks)
2. [Comparative Study of deep learning software frameworks](#comparative-study-of-deep-learning-software-frameworks)
3. [DLBench: a comprehensive experimental evaluation of deep learning frameworks]
(#dlbench-a-comprehensive-experimental-evaluation-of-deep-learning-frameworks)

## Performance comparison of deep learning frameworks
Published in [dergipark](https://dergipark.org.tr/en/download/article-file/1201877)
### Summary:
- This study is very closely linked to our work. The project aims to compare between Caffe, Caffe2, MXnet, Tensorflow, torch, Theano, and Keras. Firstly, the paper lists down the different features of each framework, and then compares between them. Initially, a table is provided to compare between various features such as- programming language, GPU support, distributed training, developed by, supported by, interface provided. This is a good starting point to compare between the various libraries.
- The study has used MNIST and GPDS datasets to compare between the different libraries, and have tinkered with different values of batch sizes and number of epochs. Maybe, we can try to take some motivation from this and try out different combinations of hyperparameters to compare between the libraries.
- Additionally, the study has noted to do all the comparison and study on the same GPU and python version, which is a good practice to follow.
- By running the model on a large dataset, the study has encountered cases of memory overflow, which is a good point to note, as it helped to test the limits of the different libraries.
- An important point to note is that, however, the study has not worked on various models, and has rather stuck to only one problem and tried to study it in depth. Also, the study has noted that various research papers have compared between the libraries, but have not stayed consistent with versions among other things, and hence they have tried to eradicate this issue.
### Key Takeaways of the paper:
- Very closely related to our work, and hence we can take a lot of inspiration from this paper.
- The paper has tried to compare between the libraries on the same GPU and python version, which is a good practice to follow.
- The paper has tried to limit its scope to only one problem, rather than various ML/DL algorithms, which might be good idea since we can study in depth.
- The researchers have tried to tinker with different hyperparameters, which we have not done yet, and possibly we can try to implement this to better understand the differences between the libraries.
- We should try to work on larger datasets, where we can possibly encounter memory overflow issues, and hence we can test the limits of the libraries.

## Comparative Study of deep learning software frameworks
Published in [arxiv](https://arxiv.org/pdf/1511.06435.pdf)
### Summary:
- The study compares between Caffe, Neon, Tensorflow, Theano, and Torch on parameters- extensibility (various DL architectures), hardware utilisation (use of CPU/GPU) and speed(both training and predicting). The study is centered around benchmarking popular DL architectures such as AlexNet, LeNet, stacked autoencoders, and LSTM.
- The study has used various datasetes of varying sizes and properties such as- ImageNet, MNIST, and IMDB dataset. 
- The researchers have tried to tinker with the hyperparameters and have presented the study on various combinations of hyperparameters along with the training time, accuracy time and GPU memory usage. 
- Have noted to compare all the libraries on the same configuration of machine and with the same version of python.
- Have noted to compare with both CPU and GPU usage on the machine. 
### Key Takeaways of the paper:
- The study is very similar to our work, and hence we can take a lot of inspiration from this paper.
- The study has worked on more advanced DL architectures, and probably we can try to implement some of them instead of vanilla ML algorithms.
- We can try to work on hyperparameters and try to compare between the libraries on various combinations of hyperparameters.

## DLBench: a comprehensive experimental evaluation of deep learning frameworks
Published in [Springer Link](https://link.springer.com/article/10.1007/s10586-021-03240-4)
### Summary:
- Aims to compare between TensorFlow, MXNet, PyTorch, Theano, Chainer, and Keras based on three DL models- CNN, Region based CNN (R-CNN), and LSTM. 
- The comparison metrics used are - training time, accuracy, convergence, resource consumption patterns. Have conducted the study on both CPU and GPU environments using different datasets. 
- The study has used six popular datasets for study- MNIST, CIFAR-10, CIFAR-100, SVHN, IMDB reviews, and VOC2012.
- For studying CPU/GPU utilisation, the study has used the metric as how frequently the CPU is utilized during the training of the deep learning models. 
- Instead of trying out by varying several hyperparameters, the study has focussed on working with the default hyperparameters which were set, and the epochs were varied. 
### Shortcomings: 
- After carefully going through source code, it is noted that the study has made use of the time module instead of using profilers provided by frameworks for performing benchmarkings.


# Review and comparative analysis of machine learning libraries for machine learning
- The research paper linked below has conducted a very similar experiment where they have compared the performance time with respect to almost the same accuracy for Neural Networks across libraries:
https://www.researchgate.net/publication/339378210_Review_and_comparative_analysis_of_machine_learning_libraries_for_machine_learning

# Kaggle Survey
We also looked at https://www.kaggle.com/kaggle-survey-2022
A Survey conducted by Kaggle which shows the preference of developers for libraries for Machine Learning. The above resource can be utilized as a parameter for ease of use and community support.

# MLModelScope: A Distributed Platform for ML Model Evaluation and Benchmarking at Scale 
Abstract: Machine Learning (ML) and Deep Learning (DL) innovations are being introduced at such a rapid pace that researchers are hard-pressed to analyze and study them. The complicated procedures for evaluating innovations, along with the lack of standard and efficient ways of specifying and provisioning ML/DL evaluation, is a major "pain point" for the community. This paper proposes MLModelScope, an open-source, framework/hardware agnostic, extensible and customizable design that enables repeatable, fair, and scalable model evaluation and benchmarking.  We implement the distributed design with support for all major frameworks and hardware, and equip it with web, command-line, and library interfaces. To demonstrate MLModelScope's capabilities we perform parallel evaluation and show how subtle changes to model evaluation pipeline affects the accuracy and HW/SW stack choices affect performance.

# Benchmarking State-of-the-Art Deep Learning Software Tools
Abstract:
Deep learning has been shown as a successful machine learning method for a variety of tasks, and its popularity results in numerous open-source deep learning software tools coming to public. Training a deep network is usually a very time-consuming process. To address the huge computational challenge in deep learning, many tools exploit hardware features such as multi-core CPUs and many-core GPUs to shorten the training and inference time. However, different tools exhibit different features and running performance when they train different types of deep networks on different hardware platforms, making it difficult for end users to select an appropriate pair of software and hardware. In this paper, we present our attempt to benchmark several state-of-the-art GPU-accelerated deep learning software tools, including Caffe, CNTK, TensorFlow, and Torch. We focus on evaluating the running time performance (i.e., speed) of these tools with three popular types of neural networks on two representative CPU platforms and three representative GPU platforms. Our contribution is two-fold. First, for end users of deep learning software tools, our benchmarking results can serve as a reference to selecting appropriate hardware platforms and software tools. Second, for developers of deep learning software tools, our in-depth analysis points out possible future directions to further optimize the running performance.
