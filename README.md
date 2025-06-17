# Data Science Cohort

Welcome to the Data Science cohort! This repository contains resources, code, and materials for a smooth learning and onboarding process.

## 🎯 Overview

This cohort is designed to help you understand and implement Machine Learning and Artificial Intellignece Models from scratch. We'll cover everything from the fundamentals of building and training your own models. You will each work on a use-case that you will do end-to-end with your teammates.

## Table of Contents

- [What is Data Science?](#what-is-data-science)
- [Where do I Start?](#where-do-i-start)
- [Training Resources](#training-resources)
  - [Tutorials](#tutorials)
- [The Data Science Toolbox](#the-data-science-toolbox)
  - [Algorithms](#algorithms)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
    - [Semi-Supervised Learning](#semi-supervised-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Data  Mining Algorithms](#data-mining-algorithms)
    - [Deep Learning Architectures](#deep-learning-architectures)
  - [General Machine Learning Packages](#general-machine-learning-packages)
  - [Deep Learning Packages](#deep-learning-packages)
    - [PyTorch Ecosystem](#pytorch-ecosystem)
    - [TensorFlow Ecosystem](#tensorflow-ecosystem)
    - [Keras Ecosystem](#keras-ecosystem)
  - [Visualization Tools](#visualization-tools)
  - [Miscellaneous Tools](#miscellaneous-tools)
- [Datasets](#datasets)


## What is Data Science?
**[`^        back to top        ^`](#awesome-data-science)**

Data Science is one of the hottest topics on the Computer and Internet farmland nowadays. People have gathered data from applications and systems until today and now is the time to analyze them. The next steps are producing suggestions from the data and creating predictions about the future. [Here](https://www.quora.com/Data-Science/What-is-data-science) you can find the biggest question for **Data Science** and hundreds of answers from experts.


| Link | Preview |
| --- | --- |
| [What is Data Science @ O'reilly](https://www.oreilly.com/ideas/what-is-data-science) | _Data scientists combine entrepreneurship with patience, the willingness to build data products incrementally, the ability to explore, and the ability to iterate over a solution. They are inherently interdisciplinary. They can tackle all aspects of a problem, from initial data collection and data conditioning to drawing conclusions. They can think outside the box to come up with new ways to view the problem, or to work with very broadly defined problems: “here’s a lot of data, what can you make from it?”_ |
| [What is Data Science @ Quora](https://www.quora.com/Data-Science/What-is-data-science) | Data Science is a combination of a number of aspects of Data such as Technology, Algorithm development, and data interference to study the data, analyse it, and find innovative solutions to difficult problems. Basically Data Science is all about Analysing data and driving for business growth by finding creative ways. |
| [The sexiest job of 21st century](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century) | _Data scientists today are akin to Wall Street “quants” of the 1980s and 1990s. In those days people with backgrounds in physics and math streamed to investment banks and hedge funds, where they could devise entirely new algorithms and data strategies. Then a variety of universities developed master’s programs in financial engineering, which churned out a second generation of talent that was more accessible to mainstream firms. The pattern was repeated later in the 1990s with search engineers, whose rarefied skills soon came to be taught in computer science programs._ |
| [Wikipedia](https://en.wikipedia.org/wiki/Data_science) | _Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from many structural and unstructured data. Data science is related to data mining, machine learning and big data._ |
| [How to Become a Data Scientist](https://www.mastersindatascience.org/careers/data-scientist/) | _Data scientists are big data wranglers, gathering and analyzing large sets of structured and unstructured data. A data scientist’s role combines computer science, statistics, and mathematics. They analyze, process, and model data then interpret the results to create actionable plans for companies and other organizations._ |
| [a very short history of #datascience](https://www.forbes.com/sites/gilpress/2013/05/28/a-very-short-history-of-data-science/) | _The story of how data scientists became sexy is mostly the story of the coupling of the mature discipline of statistics with a very young one--computer science.  The term “Data Science” has emerged only recently to specifically designate a new profession that is expected to make sense of the vast stores of big data. But making sense of data has a long history and has been discussed by scientists, statisticians, librarians, computer scientists and others for years. The following timeline traces the evolution of the term “Data Science” and its use, attempts to define it, and related terms._ |
|[Software Development Resources for Data Scientists](https://www.rstudio.com/blog/software-development-resources-for-data-scientists/)|_Data scientists concentrate on making sense of data through exploratory analysis, statistics, and models. Software developers apply a separate set of knowledge with different tools. Although their focus may seem unrelated, data science teams can benefit from adopting software development best practices. Version control, automated testing, and other dev skills help create reproducible, production-ready code and tools._|
|[Data Scientist Roadmap](https://www.scaler.com/blog/how-to-become-a-data-scientist/)|_Data science is an excellent career choice in today’s data-driven world where approx 328.77 million terabytes of data are generated daily. And this number is only increasing day by day, which in turn increases the demand for skilled data scientists who can utilize this data to drive business growth._|
|[Navigating Your Path to Becoming a Data Scientist](https://www.appliedaicourse.com/blog/how-to-become-a-data-scientist/)|_Data science is one of the most in-demand careers today. With businesses increasingly relying on data to make decisions, the need for skilled data scientists has grown rapidly. Whether it’s tech companies, healthcare organizations, or even government institutions, data scientists play a crucial role in turning raw data into valuable insights. But how do you become a data scientist, especially if you’re just starting out? _|

## Where do I Start?
**[`^        back to top        ^`](#awesome-data-science)**

While not strictly necessary, having a programming language is a crucial skill to be effective as a data scientist. Currently, the most popular language is _Python_, closely followed by _R_. Python is a general-purpose scripting language that sees applications in a wide variety of fields. R is a domain-specific language for statistics, which contains a lot of common statistics tools out of the box.

[Python](https://python.org/) is by far the most popular language in science, due in no small part to the ease at which it can be used and the vibrant ecosystem of user-generated packages. To install packages, there are two main methods: Pip (invoked as `pip install`), the package manager that comes bundled with Python, and [Anaconda](https://www.anaconda.com) (invoked as `conda install`), a powerful package manager that can install packages for Python, R, and can download executables like Git. 

Unlike R, Python was not built from the ground up with data science in mind, but there are plenty of third party libraries to make up for this. A much more exhaustive list of packages can be found later in this document, but these four packages are a good set of choices to start your data science journey with: [Scikit-Learn](https://scikit-learn.org/stable/index.html) is a general-purpose data science package which implements the most popular algorithms - it also includes rich documentation, tutorials, and examples of the models it implements. Even if you prefer to write your own implementations, Scikit-Learn is a valuable reference to the nuts-and-bolts behind many of the common algorithms you'll find. With [Pandas](https://pandas.pydata.org/), one can collect and analyze their data into a convenient table format. [Numpy](https://numpy.org/) provides very fast tooling for mathematical operations, with a focus on vectors and matrices. [Seaborn](https://seaborn.pydata.org/), itself based on the [Matplotlib](https://matplotlib.org/) package, is a quick way to generate beautiful visualizations of your data, with many good defaults available out of the box, as well as a gallery showing how to produce many common visualizations of your data.

 When embarking on your journey to becoming a data scientist, the choice of language isn't particularly important, and both Python and R have their pros and cons. Pick a language you like, and check out one of the [Free courses](#free-courses) we've listed below!
 
## Real World
**[`^        back to top        ^`](#awesome-data-science)**

Data science is a powerful tool that is utilized in various fields to solve real-world problems by extracting insights and patterns from complex data.

### Disaster
**[`^        back to top        ^`](#awesome-data-science)**

- [deprem-ml](https://huggingface.co/deprem-ml) [AYA: Açık Yazılım Ağı](https://linktr.ee/acikyazilimagi) (+25k developers) is trying to help disaster response using artificial intelligence. Everything is open-sourced [afet.org](https://afet.org). 

 

## Training Resources
**[`^        back to top        ^`](#awesome-data-science)**

How do you learn data science? By doing data science, of course! Okay, okay - that might not be particularly helpful when you're first starting out. In this section, we've listed some learning resources, in rough order from least to greatest commitment - [Tutorials](#tutorials), [Massively Open Online Courses (MOOCs)](#moocs), [Intensive Programs](#intensive-programs), and [Colleges](#colleges).


### Tutorials
**[`^        back to top        ^`](#awesome-data-science)**

- [1000 Data Science Projects](https://cloud.blobcity.com/#/ps/explore) you can run on the browser with IPython.
- [#tidytuesday](https://github.com/rfordatascience/tidytuesday) A weekly data project aimed at the R ecosystem.
- [Data science your way](https://github.com/jadianes/data-science-your-way)
- [PySpark Cheatsheet](https://github.com/kevinschaich/pyspark-cheatsheet)
- [Machine Learning, Data Science and Deep Learning with Python ](https://www.manning.com/livevideo/machine-learning-data-science-and-deep-learning-with-python)
- [Your Guide to Latent Dirichlet Allocation](https://medium.com/@lettier/how-does-lda-work-ill-explain-using-emoji-108abf40fa7d)
- [Tutorials of source code from the book Genetic Algorithms with Python by Clinton Sheppard](https://github.com/handcraftsman/GeneticAlgorithmsWithPython)
- [Tutorials to get started on signal processing for machine learning](https://github.com/jinglescode/python-signal-processing)
- [Realtime deployment](https://www.microprediction.com/python-1) Tutorial on Python time-series model deployment.
- [Python for Data Science: A Beginner’s Guide](https://learntocodewith.me/posts/python-for-data-science/)
- [Minimum Viable Study Plan for Machine Learning Interviews](https://github.com/khangich/machine-learning-interview)
- [Understand and Know Machine Learning Engineering by Building Solid Projects](http://mlzoomcamp.com/)
- [12 free Data Science projects to practice Python and Pandas](https://www.datawars.io/articles/12-free-data-science-projects-to-practice-python-and-pandas)
- [Best CV/Resume for Data Science Freshers](https://enhancv.com/resume-examples/data-scientist/)
- [Understand Data Science Course in Java](https://www.alter-solutions.com/articles/java-data-science)
- [Data Analytics Interview Questions (Beginner to Advanced)](https://www.appliedaicourse.com/blog/data-analytics-interview-questions/)
- [Top 100+ Data Science Interview Questions and Answers](https://www.appliedaicourse.com/blog/data-science-interview-questions/)


## The Data Science Toolbox
**[`^        back to top        ^`](#awesome-data-science)**

This section is a collection of packages, tools, algorithms, and other useful items in the data science world.

### Algorithms
**[`^        back to top        ^`](#awesome-data-science)**

These are some Machine Learning and Data Mining algorithms and models help you to understand your data and derive meaning from it.

#### Three kinds of Machine Learning Systems

- Based on training with human supervision
- Based on learning incrementally on fly
- Based on data points comparison and pattern detection

### Comparison
- [datacompy](https://github.com/capitalone/datacompy) - DataComPy is a package to compare two Pandas DataFrames.
  
#### Supervised Learning

- [Regression](https://en.wikipedia.org/wiki/Regression)
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Stepwise Regression](https://en.wikipedia.org/wiki/Stepwise_regression)
- [Multivariate Adaptive Regression Splines](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline)
- [Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
- [Locally Estimated Scatterplot Smoothing](https://en.wikipedia.org/wiki/Local_regression)
- Classification
  - [k-nearest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  - [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)
  - [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree)
  - [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm)
  - [C4.5 algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm)
- [Ensemble Learning](https://scikit-learn.org/stable/modules/ensemble.html)
  - [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
  - [Stacking](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python)
  - [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
  - [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
  - [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)

#### Unsupervised Learning
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering)
  - [Hierchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
  - [k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
  - [Density-based clustering](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
  - [Fuzzy clustering](https://en.wikipedia.org/wiki/Fuzzy_clustering)
  - [Mixture models](https://en.wikipedia.org/wiki/Mixture_model)
- [Dimension Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
  - [Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
  - [t-SNE; t-distributed Stochastic Neighbor Embedding](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
  - [Factor Analysis](https://scikit-learn.org/stable/modules/decomposition.html#factor-analysis)
  - [Latent Dirichlet Allocation (LDA)](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda)
- [Neural Networks](https://en.wikipedia.org/wiki/Neural_network)
- [Self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map)
- [Adaptive resonance theory](https://en.wikipedia.org/wiki/Adaptive_resonance_theory)
- [Hidden Markov Models (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model)

#### Semi-Supervised Learning

- S3VM
- [Clustering](https://en.wikipedia.org/wiki/Weak_supervision#Cluster_assumption)
- [Generative models](https://en.wikipedia.org/wiki/Weak_supervision#Generative_models)
- [Low-density separation](https://en.wikipedia.org/wiki/Weak_supervision#Low-density_separation)
- [Laplacian regularization](https://en.wikipedia.org/wiki/Weak_supervision#Laplacian_regularization)
- [Heuristic approaches](https://en.wikipedia.org/wiki/Weak_supervision#Heuristic_approaches)

#### Reinforcement Learning

- [Q Learning](https://en.wikipedia.org/wiki/Q-learning)
- [SARSA (State-Action-Reward-State-Action) algorithm](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action)
- [Temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning#:~:text=Temporal%20difference%20(TD)%20learning%20refers,estimate%20of%20the%20value%20function.)

#### Data Mining Algorithms

- [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm)
- [k-Means](https://en.wikipedia.org/wiki/K-means_clustering)
- [SVM (Support Vector Machine)](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Apriori](https://en.wikipedia.org/wiki/Apriori_algorithm)
- [EM (Expectation-Maximization)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- [PageRank](https://en.wikipedia.org/wiki/PageRank)
- [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
- [KNN (K-Nearest Neighbors)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [CART (Classification and Regression Trees)](https://en.wikipedia.org/wiki/Decision_tree_learning)



#### Deep Learning architectures

- [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network)
- [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine)
- [Autoencoder](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [Generative Adversarial Network (GAN)](https://developers.google.com/machine-learning/gan/gan_structure)
- [Self-Organized Maps](https://en.wikipedia.org/wiki/Self-organizing_map)
- [Transformer](https://www.tensorflow.org/text/tutorials/transformer)
- [Conditional Random Field (CRF)](https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776)
- [ML System Designs)](https://www.evidentlyai.com/ml-system-design)

### General Machine Learning Packages
**[`^        back to top        ^`](#awesome-data-science)**

* [scikit-learn](https://scikit-learn.org/)
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn)
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys)
* [scikit-feature](https://github.com/jundongl/scikit-feature)
* [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate)
* [seqlearn](https://github.com/larsmans/seqlearn)
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes)
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
* [sklearn-deap](https://github.com/rsteca/sklearn-deap)
* [sigopt_sklearn](https://github.com/sigopt/sigopt-sklearn)
* [sklearn-evaluation](https://github.com/edublancas/sklearn-evaluation)
* [scikit-image](https://github.com/scikit-image/scikit-image)
* [scikit-opt](https://github.com/guofei9987/scikit-opt)
* [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs)
* [pystruct](https://github.com/pystruct/pystruct)
* [Shogun](https://www.shogun-toolbox.org/)
* [xLearn](https://github.com/aksnzhy/xlearn)
* [cuML](https://github.com/rapidsai/cuml)
* [causalml](https://github.com/uber/causalml)
* [mlpack](https://github.com/mlpack/mlpack)
* [MLxtend](https://github.com/rasbt/mlxtend)
* [modAL](https://github.com/modAL-python/modAL)
* [Sparkit-learn](https://github.com/lensacom/sparkit-learn)
* [hyperlearn](https://github.com/danielhanchen/hyperlearn)
* [dlib](https://github.com/davisking/dlib)
* [imodels](https://github.com/csinva/imodels)
* [RuleFit](https://github.com/christophM/rulefit)
* [pyGAM](https://github.com/dswah/pyGAM)
* [Deepchecks](https://github.com/deepchecks/deepchecks)
* [scikit-survival](https://scikit-survival.readthedocs.io/en/stable)
* [interpretable](https://pypi.org/project/interpretable)
* [XGBoost](https://github.com/dmlc/xgboost)
* [LightGBM](https://github.com/microsoft/LightGBM)
* [CatBoost](https://github.com/catboost/catboost)
* [PerpetualBooster](https://github.com/perpetual-ml/perpetual)
* [JAX](https://github.com/google/jax)

### Deep Learning Packages

#### PyTorch Ecosystem
* [PyTorch](https://github.com/pytorch/pytorch)
* [torchvision](https://github.com/pytorch/vision)
* [torchtext](https://github.com/pytorch/text)
* [torchaudio](https://github.com/pytorch/audio)
* [ignite](https://github.com/pytorch/ignite)
* [PyTorchNet](https://github.com/pytorch/tnt)
* [PyToune](https://github.com/GRAAL-Research/poutyne)
* [skorch](https://github.com/skorch-dev/skorch)
* [PyVarInf](https://github.com/ctallec/pyvarinf)
* [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
* [GPyTorch](https://github.com/cornellius-gp/gpytorch)
* [pyro](https://github.com/pyro-ppl/pyro)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular)
* [Yolov3](https://github.com/ultralytics/yolov3)
* [Yolov5](https://github.com/ultralytics/yolov5)
* [Yolov8](https://github.com/ultralytics/ultralytics)

#### TensorFlow Ecosystem
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [TensorLayer](https://github.com/tensorlayer/TensorLayer)
* [TFLearn](https://github.com/tflearn/tflearn)
* [Sonnet](https://github.com/deepmind/sonnet)
* [tensorpack](https://github.com/tensorpack/tensorpack)
* [TRFL](https://github.com/deepmind/trfl)
* [Polyaxon](https://github.com/polyaxon/polyaxon)
* [NeuPy](https://github.com/itdxer/neupy)
* [tfdeploy](https://github.com/riga/tfdeploy)
* [tensorflow-upstream](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)
* [TensorFlow Fold](https://github.com/tensorflow/fold)
* [tensorlm](https://github.com/batzner/tensorlm)
* [TensorLight](https://github.com/bsautermeister/tensorlight)
* [Mesh TensorFlow](https://github.com/tensorflow/mesh)
* [Ludwig](https://github.com/ludwig-ai/ludwig)
* [TF-Agents](https://github.com/tensorflow/agents)
* [TensorForce](https://github.com/tensorforce/tensorforce)

#### Keras Ecosystem

* [Keras](https://keras.io)
* [keras-contrib](https://github.com/keras-team/keras-contrib)
* [Hyperas](https://github.com/maxpumperla/hyperas)
* [Elephas](https://github.com/maxpumperla/elephas)
* [Hera](https://github.com/keplr-io/hera)
* [Spektral](https://github.com/danielegrattarola/spektral)
* [qkeras](https://github.com/google/qkeras)
* [keras-rl](https://github.com/keras-rl/keras-rl)
* [Talos](https://github.com/autonomio/talos)

#### Visualization Tools
**[`^        back to top        ^`](#awesome-data-science)**

- [altair](https://altair-viz.github.io/)
- [amcharts](https://www.amcharts.com/)
- [anychart](https://www.anychart.com/)
- [bokeh](https://bokeh.org/)
- [Comet](https://www.comet.com/site/products/ml-experiment-tracking/?utm_source=awesome-datascience)
- [slemma](https://slemma.com/)
- [cartodb](https://cartodb.github.io/odyssey.js/)
- [Cube](https://square.github.io/cube/)
- [d3plus](https://d3plus.org/)
- [Data-Driven Documents(D3js)](https://d3js.org/)
- [dygraphs](https://dygraphs.com/)
- [exhibit](https://www.simile-widgets.org/exhibit/)
- [gephi](https://gephi.org/)
- [ggplot2](https://ggplot2.tidyverse.org/)
- [Glue](http://docs.glueviz.org/en/latest/index.html)
- [Google Chart Gallery](https://developers.google.com/chart/interactive/docs/gallery)
- [highcarts](https://www.highcharts.com/)
- [import.io](https://www.import.io/)
- [Matplotlib](https://matplotlib.org/)
- [nvd3](https://nvd3.org/)
- [Netron](https://github.com/lutzroeder/netron)
- [Openrefine](https://openrefine.org/)
- [plot.ly](https://plot.ly/)
- [raw](https://rawgraphs.io)
- [Resseract Lite](https://github.com/abistarun/resseract-lite)
- [Seaborn](https://seaborn.pydata.org/)
- [techanjs](https://techanjs.org/)
- [Timeline](https://timeline.knightlab.com/)
- [variancecharts](https://variancecharts.com/index.html)
- [vida](https://vida.io/)
- [vizzu](https://github.com/vizzuhq/vizzu-lib)
- [Wrangler](http://vis.stanford.edu/wrangler/)
- [r2d3](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [NetworkX](https://networkx.org/)
- [Redash](https://redash.io/)
- [C3](https://c3js.org/)
- [TensorWatch](https://github.com/microsoft/tensorwatch)
- [geomap](https://pypi.org/project/geomap/)
- [Dash](https://plotly.com/dash/)

### Miscellaneous Tools
**[`^        back to top        ^`](#awesome-data-science)**

| Link | Description |
| --- | --- |
| [The Data Science Lifecycle Process](https://github.com/dslp/dslp) | The Data Science Lifecycle Process is a process for taking data science teams from Idea to Value repeatedly and sustainably. The process is documented in this repo  |
| [Data Science Lifecycle Template Repo](https://github.com/dslp/dslp-repo-template) | Template repository for data science lifecycle project  |
| [RexMex](https://github.com/AstraZeneca/rexmex) | A general purpose recommender metrics library for fair evaluation.  |
| [ChemicalX](https://github.com/AstraZeneca/chemicalx) | A PyTorch based deep learning library for drug pair scoring.  |
| [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) | Representation learning on dynamic graphs.  |
| [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) | A graph sampling library for NetworkX with a Scikit-Learn like API.  |
| [Karate Club](https://github.com/benedekrozemberczki/karateclub) | An unsupervised machine learning extension library for NetworkX with a Scikit-Learn like API. |
| [ML Workspace](https://github.com/ml-tooling/ml-workspace) | All-in-one web-based IDE for machine learning and data science. The workspace is deployed as a Docker container and is preloaded with a variety of popular data science libraries (e.g., Tensorflow, PyTorch) and dev tools (e.g., Jupyter, VS Code) |
| [Neptune.ai](https://neptune.ai) | Community-friendly platform supporting data scientists in creating and sharing machine learning models. Neptune facilitates teamwork, infrastructure management, models comparison and reproducibility. |
| [steppy](https://github.com/minerva-ml/steppy) | Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces very simple interface that enables clean machine learning pipeline design. |
| [steppy-toolkit](https://github.com/minerva-ml/steppy-toolkit) | Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective. |
| [Datalab from Google](https://cloud.google.com/datalab/docs/) | easily explore, visualize, analyze, and transform data using familiar languages, such as Python and SQL, interactively. |
| [Hortonworks Sandbox](https://www.cloudera.com/downloads/hortonworks-sandbox.html) | is a personal, portable Hadoop environment that comes with a dozen interactive Hadoop tutorials. |
| [R](https://www.r-project.org/) | is a free software environment for statistical computing and graphics. |
| [Tidyverse](https://www.tidyverse.org/) | is an opinionated collection of R packages designed for data science. All packages share an underlying design philosophy, grammar, and data structures. |
| [RStudio](https://www.rstudio.com) | IDE – powerful user interface for R. It’s free and open source, and works on Windows, Mac, and Linux. |
| [Python - Pandas - Anaconda](https://www.anaconda.com) | Completely free enterprise-ready Python distribution for large-scale data processing, predictive analytics, and scientific computing |
| [Pandas GUI](https://github.com/adrotog/PandasGUI) | Pandas GUI |
| [Scikit-Learn](https://scikit-learn.org/stable/) | Machine Learning in Python |
| [NumPy](https://numpy.org/) | NumPy is fundamental for scientific computing with Python. It supports large, multi-dimensional arrays and matrices and includes an assortment of high-level mathematical functions to operate on these arrays. |
| [Vaex](https://vaex.io/) | Vaex is a Python library that allows you to visualize large datasets and calculate statistics at high speeds. |
| [SciPy](https://scipy.org/) | SciPy works with NumPy arrays and provides efficient routines for numerical integration and optimization. |
| [Data Science Toolbox](https://www.coursera.org/learn/data-scientists-tools) | Coursera Course |
| [Data Science Toolbox](https://datasciencetoolbox.org/) | Blog |
| [Wolfram Data Science Platform](https://www.wolfram.com/data-science-platform/) | Take numerical, textual, image, GIS or other data and give it the Wolfram treatment, carrying out a full spectrum of data science analysis and visualization and automatically generate rich interactive reports—all powered by the revolutionary knowledge-based Wolfram Language. |
| [Datadog](https://www.datadoghq.com/) | Solutions, code, and devops for high-scale data science. |
| [Variance](https://variancecharts.com/) | Build powerful data visualizations for the web without writing JavaScript |
| [Kite Development Kit](http://kitesdk.org/docs/current/index.html) | The Kite Software Development Kit (Apache License, Version 2.0), or Kite for short, is a set of libraries, tools, examples, and documentation focused on making it easier to build systems on top of the Hadoop ecosystem. |
| [Domino Data Labs](https://www.dominodatalab.com) | Run, scale, share, and deploy your models — without any infrastructure or setup. |
| [Apache Flink](https://flink.apache.org/) | A platform for efficient, distributed, general-purpose data processing. |
| [Apache Hama](https://hama.apache.org/) | Apache Hama is an Apache Top-Level open source project, allowing you to do advanced analytics beyond MapReduce. |
| [Weka](https://ml.cms.waikato.ac.nz/weka/index.html) | Weka is a collection of machine learning algorithms for data mining tasks. |
| [Octave](https://www.gnu.org/software/octave/) | GNU Octave is a high-level interpreted language, primarily intended for numerical computations.(Free Matlab) |
| [Apache Spark](https://spark.apache.org/) | Lightning-fast cluster computing |
| [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) | a service for exposing Apache Spark analytics jobs and machine learning models as realtime, batch or reactive web services. |
| [Data Mechanics](https://www.datamechanics.co) | A data science and engineering platform making Apache Spark more developer-friendly and cost-effective. |
| [Caffe](https://caffe.berkeleyvision.org/) | Deep Learning Framework |
| [Torch](http://torch.ch/) | A SCIENTIFIC COMPUTING FRAMEWORK FOR LUAJIT |
| [Nervana's python based Deep Learning Framework](https://github.com/NervanaSystems/neon) | Intel® Nervana™ reference deep learning framework committed to best performance on all hardware. |
| [Skale](https://github.com/skale-me/skale) | High performance distributed data processing in NodeJS |
| [Aerosolve](https://airbnb.io/aerosolve/) | A machine learning package built for humans. |
| [Intel framework](https://github.com/intel/idlf) | Intel® Deep Learning Framework |
| [Datawrapper](https://www.datawrapper.de/) | An open source data visualization platform helping everyone to create simple, correct and embeddable charts. Also at [github.com](https://github.com/datawrapper/datawrapper) |
| [Tensor Flow](https://www.tensorflow.org/) | TensorFlow is an Open Source Software Library for Machine Intelligence |
| [Natural Language Toolkit](https://www.nltk.org/) | An introductory yet powerful toolkit for natural language processing and classification |
| [Annotation Lab](https://www.johnsnowlabs.com/annotation-lab/) | Free End-to-End No-Code platform for text annotation and DL model training/tuning. Out-of-the-box support for Named Entity Recognition, Classification, Relation extraction and Assertion Status Spark NLP models. Unlimited support for users, teams, projects, documents. |
| [nlp-toolkit for node.js](https://www.npmjs.com/package/nlp-toolkit) | This module covers some basic nlp principles and implementations. The main focus is performance. When we deal with sample or training data in nlp, we quickly run out of memory. Therefore every implementation in this module is written as stream to only hold that data in memory that is currently processed at any step. |
| [Julia](https://julialang.org) | high-level, high-performance dynamic programming language for technical computing |
| [IJulia](https://github.com/JuliaLang/IJulia.jl) | a Julia-language backend combined with the Jupyter interactive environment |
| [Apache Zeppelin](https://zeppelin.apache.org/) | Web-based notebook that enables data-driven, interactive data analytics and collaborative documents with SQL, Scala and more  |
| [Featuretools](https://github.com/alteryx/featuretools) | An open source framework for automated feature engineering written in python |
| [Optimus](https://github.com/hi-primus/optimus) | Cleansing, pre-processing, feature engineering, exploratory data analysis and easy ML with PySpark backend.  |
| [Albumentations](https://github.com/albumentations-team/albumentations) | А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, and detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops. |
| [DVC](https://github.com/iterative/dvc) | An open-source data science version control system. It helps track, organize and make data science projects reproducible. In its very basic scenario it helps version control and share large data and model files. |
| [Lambdo](https://github.com/asavinov/lambdo) | is a workflow engine that significantly simplifies data analysis by combining in one analysis pipeline (i) feature engineering and machine learning (ii) model training and prediction (iii) table population and column evaluation. |
| [Feast](https://github.com/feast-dev/feast) | A feature store for the management, discovery, and access of machine learning features. Feast provides a consistent view of feature data for both model training and model serving. |
| [Polyaxon](https://github.com/polyaxon/polyaxon) | A platform for reproducible and scalable machine learning and deep learning. |
| [UBIAI](https://ubiai.tools) | Easy-to-use text annotation tool for teams with most comprehensive auto-annotation features. Supports NER, relations and document classification as well as OCR annotation for invoice labeling |
| [Trains](https://github.com/allegroai/clearml) | Auto-Magical Experiment Manager, Version Control & DevOps for AI |
| [Hopsworks](https://github.com/logicalclocks/hopsworks) | Open-source data-intensive machine learning platform with a feature store. Ingest and manage features for both online (MySQL Cluster)  and offline (Apache Hive) access, train and serve models at scale. |
| [MindsDB](https://github.com/mindsdb/mindsdb) | MindsDB is an Explainable AutoML framework for developers. With MindsDB you can build, train and use state of the art ML models in as simple as one line of code. |
| [Lightwood](https://github.com/mindsdb/lightwood) | A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with an objective to build predictive models with one line of code. |
| [AWS Data Wrangler](https://github.com/awslabs/aws-data-wrangler) | An open-source Python package that extends the power of Pandas library to AWS connecting DataFrames and AWS data related services (Amazon Redshift, AWS Glue, Amazon Athena, Amazon EMR, etc). |
| [Amazon Rekognition](https://aws.amazon.com/rekognition/) | AWS Rekognition is a service that lets developers working with Amazon Web Services add image analysis to their applications. Catalog assets, automate workflows, and extract meaning from your media and applications.|
| [Amazon Textract](https://aws.amazon.com/textract/) | Automatically extract printed text, handwriting, and data from any document. |
| [Amazon Lookout for Vision](https://aws.amazon.com/lookout-for-vision/) | Spot product defects using computer vision to automate quality inspection. Identify missing product components, vehicle and structure damage, and irregularities for comprehensive quality control.|
| [Amazon CodeGuru](https://aws.amazon.com/codeguru/) | Automate code reviews and optimize application performance with ML-powered recommendations.|
| [CML](https://github.com/iterative/cml) | An open source toolkit for using continuous integration in data science projects. Automatically train and test models in production-like environments with GitHub Actions & GitLab CI, and autogenerate visual reports on pull/merge requests. |
| [Dask](https://dask.org/) | An open source Python library to painlessly transition your analytics code to distributed computing systems (Big Data) |
| [Statsmodels](https://www.statsmodels.org/stable/index.html) | A Python-based inferential statistics, hypothesis testing and regression framework |
| [Gensim](https://radimrehurek.com/gensim/) | An open-source library for topic modeling of natural language text |
| [spaCy](https://spacy.io/) | A performant natural language processing toolkit |
| [Grid Studio](https://github.com/ricklamers/gridstudio) | Grid studio is a web-based spreadsheet application with full integration of the Python programming language. |
|[Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook)|Python Data Science Handbook: full text in Jupyter Notebooks|
| [Shapley](https://github.com/benedekrozemberczki/shapley) | A data-driven framework to quantify the value of classifiers in a machine learning ensemble.  |
| [DAGsHub](https://dagshub.com) | A platform built on open source tools for data, model and pipeline management.  |
| [Deepnote](https://deepnote.com) | A new kind of data science notebook. Jupyter-compatible, with real-time collaboration and running in the cloud. |
| [Valohai](https://valohai.com) | An MLOps platform that handles machine orchestration, automatic reproducibility and deployment. |
| [PyMC3](https://docs.pymc.io/) | A Python Library for Probabalistic Programming (Bayesian Inference and Machine Learning) |
| [PyStan](https://pypi.org/project/pystan/) | Python interface to Stan (Bayesian inference and modeling) |
| [hmmlearn](https://pypi.org/project/hmmlearn/) | Unsupervised learning and inference of Hidden Markov Models |
| [Chaos Genius](https://github.com/chaos-genius/chaos_genius/) | ML powered analytics engine for outlier/anomaly detection and root cause analysis |
| [Nimblebox](https://nimblebox.ai/) | A full-stack MLOps platform designed to help data scientists and machine learning practitioners around the world discover, create, and launch multi-cloud apps from their web browser. |
| [Towhee](https://github.com/towhee-io/towhee) | A Python library that helps you encode your unstructured data into embeddings. |
| [LineaPy](https://github.com/LineaLabs/lineapy) | Ever been frustrated with cleaning up long, messy Jupyter notebooks? With LineaPy, an open source Python library, it takes as little as two lines of code to transform messy development code into production pipelines. |
| [envd](https://github.com/tensorchord/envd) | 🏕️ machine learning development environment for data science and AI/ML engineering teams |
| [Explore Data Science Libraries](https://kandi.openweaver.com/explore/data-science) | A search engine 🔎 tool to discover & find a curated list of popular & new libraries, top authors, trending project kits, discussions, tutorials & learning resources |
| [MLEM](https://github.com/iterative/mlem) | 🐶 Version and deploy your ML models following GitOps principles |
| [MLflow](https://mlflow.org/) | MLOps framework for managing ML models across their full lifecycle |
| [cleanlab](https://github.com/cleanlab/cleanlab) | Python library for data-centric AI and automatically detecting various issues in ML datasets |
| [AutoGluon](https://github.com/awslabs/autogluon) | AutoML to easily produce accurate predictions for image, text, tabular, time-series, and multi-modal data |
| [Arize AI](https://arize.com/) | Arize AI community tier observability tool for monitoring machine learning models in production and root-causing issues such as data quality and performance drift. |
| [Aureo.io](https://aureo.io) | Aureo.io is a low-code platform that focuses on building artificial intelligence. It provides users with the capability to create pipelines, automations and integrate them with artificial intelligence models – all with their basic data. |
| [ERD Lab](https://www.erdlab.io/) | Free cloud based entity relationship diagram (ERD) tool made for developers.
| [Arize-Phoenix](https://docs.arize.com/phoenix) | MLOps in a notebook - uncover insights, surface problems, monitor, and fine tune your models. |
| [Comet](https://github.com/comet-ml/comet-examples) | An MLOps platform with experiment tracking, model production management, a model registry, and full data lineage to support your ML workflow from training straight through to production. |
| [Opik](https://github.com/comet-ml/opik) | Evaluate, test, and ship LLM applications across your dev and production lifecycles. |
| [Synthical](https://synthical.com) | AI-powered collaborative environment for research. Find relevant papers, create collections to manage bibliography, and summarize content — all in one place |
| [teeplot](https://github.com/mmore500/teeplot) | Workflow tool to automatically organize data visualization output |
| [Streamlit](https://github.com/streamlit/streamlit) | App framework for Machine Learning and Data Science projects |
| [Gradio](https://github.com/gradio-app/gradio) | Create customizable UI components around machine learning models |
| [Weights & Biases](https://github.com/wandb/wandb) | Experiment tracking, dataset versioning, and model management |
| [DVC](https://github.com/iterative/dvc) | Open-source version control system for machine learning projects |
| [Optuna](https://github.com/optuna/optuna) | Automatic hyperparameter optimization software framework |
| [Ray Tune](https://github.com/ray-project/ray) | Scalable hyperparameter tuning library |
| [Apache Airflow](https://github.com/apache/airflow) | Platform to programmatically author, schedule, and monitor workflows |
| [Prefect](https://github.com/PrefectHQ/prefect) | Workflow management system for modern data stacks |
| [Kedro](https://github.com/kedro-org/kedro) | Open-source Python framework for creating reproducible, maintainable data science code |
| [Hamilton](https://github.com/dagworks-inc/hamilton) | Lightweight library to author and manage reliable data transformations |
| [SHAP](https://github.com/slundberg/shap) | Game theoretic approach to explain the output of any machine learning model |
| [LIME](https://github.com/marcotcr/lime) | Explaining the predictions of any machine learning classifier |
| [flyte](https://github.com/flyteorg/flyte) | Workflow automation platform for machine learning |
| [dbt](https://github.com/dbt-labs/dbt-core) | Data build tool |
| [SHAP](https://github.com/slundberg/shap) | Game theoretic approach to explain the output of any machine learning model |
| [LIME](https://github.com/marcotcr/lime) | Explaining the predictions of any machine learning classifier |
| [zasper](https://github.com/zasper-io/zasper) | Supercharged IDE for Data Science |
| [skrub](https://github.com/skrub-data/skrub/) | A Python library to ease preprocessing and feature engineering for tabular machine learning |

### Datasets
**[`^        back to top        ^`](#awesome-data-science)**

- [Academic Torrents](https://academictorrents.com/)
- [ADS-B Exchange](https://www.adsbexchange.com/data-samples/) - Specific datasets for aircraft and Automatic Dependent Surveillance-Broadcast (ADS-B) sources.
- [hadoopilluminated.com](https://hadoopilluminated.com/hadoop_illuminated/Public_Bigdata_Sets.html)
- [data.gov](https://catalog.data.gov/dataset) - The home of the U.S. Government's open data
- [United States Census Bureau](https://www.census.gov/)
- [usgovxml.com](https://usgovxml.com/)
- [enigma.com](https://enigma.com/) - Navigate the world of public data - Quickly search and analyze billions of public records published by governments, companies and organizations.
- [datahub.io](https://datahub.io/)
- [aws.amazon.com/datasets](https://aws.amazon.com/datasets/)
- [datacite.org](https://datacite.org/)
- [The official portal for European data](https://data.europa.eu/en)
- [NASDAQ:DATA](https://data.nasdaq.com/) - Nasdaq Data Link A premier source for financial, economic and alternative datasets.
- [figshare.com](https://figshare.com/)
- [GeoLite Legacy Downloadable Databases](https://dev.maxmind.com/geoip)
- [Quora's Big Datasets Answer](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
- [Public Big Data Sets](https://hadoopilluminated.com/hadoop_illuminated/Public_Bigdata_Sets.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [A Deep Catalog of Human Genetic Variation](https://www.internationalgenome.org/data)
- [A community-curated database of well-known people, places, and things](https://developers.google.com/freebase/)
- [Google Public Data](https://www.google.com/publicdata/directory)
- [World Bank Data](https://data.worldbank.org/)
- [NYC Taxi data](https://chriswhong.github.io/nyctaxi/)
- [Open Data Philly](https://www.opendataphilly.org/) Connecting people with data for Philadelphia
- [grouplens.org](https://grouplens.org/datasets/) Sample movie (with ratings), book and wiki datasets
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/) - contains data sets good for machine learning
- [research-quality data sets](https://web.archive.org/web/20150320022752/https://bitly.com/bundles/hmason/1) by [Hilary Mason](https://web.archive.org/web/20150501033715/https://bitly.com/u/hmason/bundles)
- [National Centers for Environmental Information](https://www.ncei.noaa.gov/)
- [ClimateData.us](https://www.climatedata.us/) (related: [U.S. Climate Resilience Toolkit](https://toolkit.climate.gov/))
- [r/datasets](https://www.reddit.com/r/datasets/)
- [MapLight](https://www.maplight.org/data-series) - provides a variety of data free of charge for uses that are freely available to the general public. Click on a data set below to learn more
- [GHDx](https://ghdx.healthdata.org/) - Institute for Health Metrics and Evaluation - a catalog of health and demographic datasets from around the world and including IHME results
- [St. Louis Federal Reserve Economic Data - FRED](https://fred.stlouisfed.org/)
- [New Zealand Institute of Economic Research – Data1850](https://data1850.nz/)
- [Open Data Sources](https://github.com/datasciencemasters/data)
- [UNICEF Data](https://data.unicef.org/)
- [undata](https://data.un.org/)
- [NASA SocioEconomic Data and Applications Center - SEDAC](https://earthdata.nasa.gov/centers/sedac-daac)
- [The GDELT Project](https://www.gdeltproject.org/)
- [Sweden, Statistics](https://www.scb.se/en/)
- [StackExchange Data Explorer](https://data.stackexchange.com) - an open source tool for running arbitrary queries against public data from the Stack Exchange network.
- [San Fransisco Government Open Data](https://datasf.org/opendata/)
- [IBM Asset Dataset](https://developer.ibm.com/exchanges/data/)
- [Open data Index](http://index.okfn.org/)
- [Public Git Archive](https://github.com/src-d/datasets/tree/master/PublicGitArchive)
- [GHTorrent](https://ghtorrent.org/)
- [Microsoft Research Open Data](https://msropendata.com/)
- [Open Government Data Platform India](https://data.gov.in/)
- [Google Dataset Search (beta)](https://datasetsearch.research.google.com/)
- [NAYN.CO Turkish News with categories](https://github.com/naynco/nayn.data)
- [Covid-19](https://github.com/datasets/covid-19)
- [Covid-19 Google](https://github.com/google-research/open-covid-19-data)
- [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)
- [5000 Images of Clothes](https://github.com/alexeygrigorev/clothing-dataset)
- [IBB Open Portal](https://data.ibb.gov.tr/en/)
- [The Humanitarian Data Exchange](https://data.humdata.org/)
- [250k+ Job Postings](https://aws.amazon.com/marketplace/pp/prodview-p2554p3tczbes) - An expanding dataset of historical job postings from Luxembourg from 2020 to today. Free with 250k+ job postings hosted on AWS Data Exchange.


Credits: https://github.com/academic/awesome-datascience?tab=readme-ov-file#datasets 

## 💻 Development Environment Setup

### System Requirements
- Python 3.11 or higher
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (recommended for local model training)
- 50GB+ free disk space

### Step-by-Step Setup Guide

1. **Install Python 3.11+**
   ```bash
   # Windows (using winget)
   winget install Python.Python.3.11

   # macOS (using Homebrew)
   brew install python@3.11

   # Linux (Ubuntu/Debian)
   sudo apt update
   sudo apt install python3.11 python3.11-venv
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv llm-env

   # Activate on Windows
   .\llm-env\Scripts\activate

   # Activate on macOS/Linux
   source llm-env/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   # Install basic requirements
   pip install --upgrade pip
   pip install torch torchvision torchaudio
   pip install transformers datasets accelerate
   pip install sentencepiece protobuf
   pip install bitsandbytes  # for 4-bit quantization
   pip install scipy numpy pandas
   ```

4. **Install CUDA (for NVIDIA GPUs)**
   - Download and install CUDA Toolkit from NVIDIA website
   - Install cuDNN for better performance
   - Verify installation:
     ```bash
     nvidia-smi
     python -c "import torch; print(torch.cuda.is_available())"
     ```



## 💡 Best Practices & Tips

### Memory-Efficient Model Loading
```python
# Best Practice: Use quantization and model offloading
def load_model_efficiently(model_name, device="cuda"):
    try:
        # 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Enable model offloading
        if device == "cuda":
            model.enable_model_cpu_offload()
            
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

### Error Handling and Logging
```python
# Best Practice: Implement comprehensive error handling
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Retry decorator for API calls
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator
```

### Tips and Tricks

1. **Caching Responses**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_model_response(prompt):
       # Your model inference here
       pass
   ```

2. **Batch Processing**
   ```python
   def process_batch(prompts, batch_size=32):
       results = []
       for i in range(0, len(prompts), batch_size):
           batch = prompts[i:i + batch_size]
           # Process batch
           results.extend(process_single_batch(batch))
       return results
   ```

3. **Memory Management**
   ```python
   import gc
   
   def clear_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

### Best Practices Checklist

1. **API Management**
   - ✅ Use environment variables for API keys
   - ✅ Implement rate limiting
   - ✅ Use connection pooling
   - ✅ Implement retry mechanisms

2. **Model Loading**
   - ✅ Use quantization (4-bit or 8-bit)
   - ✅ Enable model offloading
   - ✅ Use mixed precision training
   - ✅ Implement gradient checkpointing

3. **Error Handling**
   - ✅ Implement comprehensive logging
   - ✅ Use try-except blocks
   - ✅ Add retry mechanisms
   - ✅ Implement fallback options

4. **Performance Optimization**
   - ✅ Use batch processing
   - ✅ Implement caching
   - ✅ Optimize memory usage
   - ✅ Use async operations where appropriate

5. **Security**
   - ✅ Never hardcode API keys
   - ✅ Implement input validation
   - ✅ Use secure connections
   - ✅ Implement proper error messages

### Common Pitfalls to Avoid

1. **Memory Issues**
   - ❌ Loading full precision models without quantization
   - ❌ Not clearing GPU memory between operations
   - ❌ Keeping unnecessary model copies in memory

2. **API Usage**
   - ❌ Not implementing rate limiting
   - ❌ Not handling API timeouts
   - ❌ Not implementing retry mechanisms

3. **Error Handling**
   - ❌ Catching all exceptions without specific handling
   - ❌ Not logging errors properly
   - ❌ Not implementing fallback options

4. **Performance**
   - ❌ Not using batch processing
   - ❌ Not implementing caching
   - ❌ Not optimizing memory usage

## ☁️ Cloud Development with Google Cloud Platform

### Getting Started with GCP (Optional)
#### Note: You can also use any other cloud provider of your choice. Here is a list of cloud providers:
https://github.com/cloudcommunity/Cloud-Free-Tier-Comparison

1. **Sign Up and Free Credits**
   - Create a Google Cloud account
   - Get $300 free credits (valid for 90 days)
   - Enable billing (required even for free tier)

2. **Install Google Cloud CLI**
   ```bash
   # Windows (using winget)
   winget install Google.CloudSDK

   # macOS (using Homebrew)
   brew install google-cloud-sdk

   # Linux
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
   sudo apt-get install apt-transport-https ca-certificates gnupg
   sudo apt-get update && sudo apt-get install google-cloud-sdk
   ```

3. **Initialize and Configure GCP**
   ```bash
   # Login to GCP
   gcloud auth login

   # Set your project
   gcloud config set project YOUR_PROJECT_ID

   # Enable required APIs
   gcloud services enable compute.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```

### Setting Up Cloud Environment

1. **Create a Virtual Machine**
   ```bash
   # Create a VM with GPU
   gcloud compute instances create llm-dev \
     --machine-type=n1-standard-4 \
     --zone=us-central1-a \
     --accelerator="type=nvidia-tesla-t4,count=1" \
     --maintenance-policy=TERMINATE \
     --image-family=debian-11-gpu \
     --image-project=debian-cloud \
     --boot-disk-size=100GB
   ```

2. **Connect to VM**
   ```bash
   # SSH into the VM
   gcloud compute ssh llm-dev --zone=us-central1-a
   ```

### Cloud Development Setup

1. **Install Dependencies on VM**
   ```bash
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install Python and dependencies
   sudo apt-get install python3.11 python3.11-venv
   python3.11 -m venv llm-env
   source llm-env/bin/activate

   # Install CUDA and cuDNN
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. **Configure Python Environment**
   ```bash
   # Install required packages
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets accelerate
   pip install google-cloud-aiplatform
   ```

### Using Google Cloud AI Platform

1. **Initialize Vertex AI**
   ```python
   from google.cloud import aiplatform

   def setup_vertex_ai():
       # Initialize Vertex AI
       aiplatform.init(
           project='your-project-id',
           location='us-central1',
           experiment='llm-experiment'
       )
       
       # Create or get endpoint
       endpoint = aiplatform.Endpoint(
           endpoint_name='projects/your-project-id/locations/us-central1/endpoints/your-endpoint-id'
       )
       return endpoint
   ```

2. **Deploy Model to Vertex AI**
   ```python
   def deploy_model_to_vertex(model_path):
       # Create model
       model = aiplatform.Model.upload(
           display_name='llm-model',
           artifact_uri=model_path,
           serving_container_image_uri='us-docker.pkg.dev/cloud-aiplatform/prediction/pytorch-gpu.1-10:latest'
       )
       
       # Deploy model
       endpoint = model.deploy(
           machine_type='n1-standard-4',
           accelerator_type='NVIDIA_TESLA_T4',
           accelerator_count=1
       )
       return endpoint
   ```

### Cost Optimization Tips

1. **Use Preemptible VMs**
   ```bash
   # Create preemptible VM (up to 80% cheaper)
   gcloud compute instances create llm-dev \
     --preemptible \
     --machine-type=n1-standard-4 \
     --zone=us-central1-a
   ```

2. **Auto-shutdown Script**
   ```bash
   # Create shutdown script
   echo '#!/bin/bash
   sudo shutdown -h now' > shutdown.sh
   chmod +x shutdown.sh

   # Add to VM creation
   gcloud compute instances create llm-dev \
     --metadata-from-file=shutdown-script=shutdown.sh
   ```

3. **Cost Monitoring**
   ```bash
   # Set budget alerts
   gcloud billing budgets create \
     --billing-account=YOUR_BILLING_ACCOUNT \
     --display-name="LLM Development Budget" \
     --budget-amount=100USD \
     --threshold-rule=percent=0.5 \
     --threshold-rule=percent=0.9
   ```

### Best Practices for Cloud Development

1. **Resource Management**
   - Use appropriate machine types
   - Implement auto-shutdown
   - Monitor resource usage
   - Clean up unused resources

2. **Data Management**
   - Use Cloud Storage for datasets
   - Implement proper backup strategies
   - Use version control for code
   - Store models in Cloud Storage

3. **Security**
   - Use service accounts
   - Implement IAM roles
   - Secure API keys
   - Enable audit logging

4. **Performance**
   - Use GPU instances when needed
   - Implement caching
   - Use batch processing
   - Optimize model size

### Cost Comparison (Approximate)

| Resource | Local Cost | GCP Cost (Free Tier) | GCP Cost (Paid) |
|----------|------------|---------------------|-----------------|
| GPU (T4) | Hardware   | $300 credits        | $0.35/hour      |
| Storage  | Free       | 5GB free            | $0.02/GB/month  |
| Network  | Free       | 1GB/day free        | $0.12/GB        |

Note: Prices may vary by region and time. Check Google Cloud pricing calculator for current rates.

## 🙏 Acknowledgments

We would like to express our sincere gratitude to the Open Source Community.
