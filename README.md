# Benchmarking ML/DL Libraries

## Problem Context

The field of machine learning is rapidly evolving, and new libraries and techniques are constantly emerg-
ing. Some of the popular machine learning libraries are TensorFlow[2], Keras[3], PyTorch[5], scikit-learn[6],
XGBoost, LightBGM, MXNet, Horovod, Lightning AI etc. Since there are so many of these libraries, it
is important for a developer to have a framework or some guidelines based on which she can decide which
library to choose to satisfy her requirements. The requirements of a developer will vary and may involve
metrics such as ease of use, performance, energy usage, flexibility, accuracy, community support, integration
with other tools, cost, deployment, security, data preprocessing tools, monitoring etc.
Our goal is to develop a framework which compares the popular open-source machine learning libraries
across a number of metrics and dimensions and suggests an appropriate library to the developer. We also
aim to develop a user-friendly web application designed to be used by Machine Learning practitioners all
over the globe for evaluating which library to choose when beginning with a project.

## Challenges with Existing Solutions

While there exist many informal sources such as blog posts and editorials[4] which compare the above
mentioned libraries on some of the above mentioned metrics, we found very few academic papers[1] on
comparative analysis and whatever we found was not comprehensive at all.
Based on our literature review, some challenges that we anticipate solving are:
1. Data availability and quality: One of the biggest challenges in benchmarking and evaluating machine
learning models is ensuring that the data used is representative, diverse, and of good quality.
2. Model complexity: The complexity of machine learning models can vary widely, and evaluating the
relative performance of different models can be challenging.
3. Reproducibility: Results that are reproducible are critical for evaluating the relative performance of
machine learning models. This requires careful documentation of experimental setups, code, and data.
4. Metrics selection: Choosing appropriate evaluation metrics is critical for benchmarking and evaluating
machine learning models. Different metrics may be appropriate for different applications, and it is
important to carefully consider the strengths and limitations of each metric.

## Targeted Features/Metrics

As part of this project, we will compare five popular open-source machine learning libraries using metrics
such as:
1. Ease of use: How easy is it to use the library, and how much documentation is available?
2. Performance: How fast and efficient are the algorithms in the library?
3. Memory Usage: How much memory is required to run a model?
4. CPU and GPU Usage: How much computation power is required by the models built using the library, and how does it compare to other
libraries in terms of computational power?
5. Scalability: How well does the library perform on large datasets, and how easy is it to scale up the
algorithms?
6. Flexibility: How flexible is the library in terms of the types of models it can build and the types of data
it can handle?
7. Accuracy: How accurate are the models built using the library, and how does it compare to other
libraries in terms of accuracy?
8. Community support: How active is the community around the library, and how well-supported is the
library in terms of bug fixes and updates?
9. Integration with other tools: How well does the library integrate with other tools and platforms, such
as Jupyter notebooks and cloud computing services?

Utilizing the results of the comparative evaluation, we intend to develop a framework that assists devel-
opers in selecting an appropriate machine learning library for their specific use case. This framework will
consider both the evaluation outcomes and the developer’s preferences, and will provide insights into the
tradeoffs associated with each library for a given use case.

## Expected Outcomes

1. Publication: A research paper for the comparative evaluation of machine learning libraries. The paper
will also discuss the methodology used to build the proposed framework for selection of machine learning
libraries.
2. Software: A user-friendly web application which takes in the developers preferences and suggests some
of the machine learning libraries to the developer along with sufficient reasoning about the framework’s
decision.
3. Future work: We intend to pursue this research in the forthcoming months to expand the scope and
depth of this project in terms of more machine learning libraries and comparison metrics.


## References

[1] Comparison of ml frameworks. Comparison of ML Frameworks - AI Wiki.  
[2] Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S.,
Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard,
M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R.,
Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I.,
Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden,
P., Wattenberg, M., Wicke, M., Yu, Y., and Zheng, X. TensorFlow: Large-scale machine learning
on heterogeneous systems. Software available from tensorflow.org.  
[3] Chollet, F., et al. Keras.  
[4] Costa, C. D. Best python libraries for machine learning and deep learning. Medium (Mar 2020).
[5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin,
Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M.,
Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An
imperative style, high-performance deep learning library. 8024–8035.  
[6] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel,
M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D.,
Brucher, M., Perrot, M., and Duchesnay, E. Scikit-learn: Machine learning in Python. Journal
of Machine Learning Research 12 (2011), 2825–2830.  


