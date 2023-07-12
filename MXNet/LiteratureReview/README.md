# Literature Review
# Table of Contents
1. [Performance comparison of deep learning frameworks](#performance-comparison-of-deep-learning-frameworks)

## Performance comparison of deep learning frameworks
Published in [dergipark](https://dergipark.org.tr/en/download/article-file/1201877)
### Key learnings and takeaways:
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

