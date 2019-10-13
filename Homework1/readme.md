第一次作业
                                                石绍松 201944762
实验要求
对所给的两个数据集，分别使用八种聚类算法对数据进行训练，将训练前后的数据进行对比，使用Normalized Mutual Information、Homogeneity以及Completeness观察各个算法的聚类效果。

实验环境
编程语言：python3.7
IDE：pycharm
实验机器：I5-3470

实验数据
1、sklearn.datasets.load_digits 包含10个类，每个类包含180个例子共1797个，维度为64，特征为0-16之间的整数。
2、sklearn.datasets.fetch_20newsgroups 包含20个类，共18846个例子，特征为文本片段。

实验过程以及实验结果
1、K-MEANS
(A)获取数据集得到标签，使用k-means算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为n_clusters =n_digits，n_init = 10,所得结果为：
Homogeneity: 0.602
Normalized Mutual Information: 0.626
Completeness: 0.650
(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用K-Means进行训练，部分参数设置为n_clusters=true_k, init='k-means++', n_init=1,所得结果为
Homogeneity: 0.485
Normalized Mutual Information: 0.498
Completeness: 0.511

2、Affinity Propagation
(A)获取数据集得到标签，使用Affinity Propagation算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为preference=-50,所得结果为:
Homogeneity: 0.964
Normalized Mutual Information: 0.640
Completeness: 0.425

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用Affinity Propagation进行训练，参数使用Affinity Propagation算法的默认参数,所得结果为：
Homogeneity: 0.885
Completeness: 0.191
Normalized Mutual Information: 0.411

3、Mean-shift
(A)获取数据集得到标签，使用Mean-shift算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为preference=-50,bandwidth=bandwidth, bin_seeding=True,所得结果为
Homogeneity: 0.009
Normalized Mutual Information: 0.048
Completeness: 0.257

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用Mean-shift进行训练，参数使用部分参数为bandwidth=bandwidth, bin_seeding=True,所得结果为：Homogeneity: 0.391
Completeness: 0.786
Normalized Mutual Information: 0.554

4、Spectral clustering
(A)获取数据集得到标签，使用Mean-shift算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为affinity="nearest_neighbors",所得结果为
Homogeneity: 0.768
Completeness: 0.874
Normalized Mutual Information:0.819

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用Spectral clustering进行训练，参数使用affinity="nearest_neighbors",所得结果为：
Homogeneity: 0.458
Completeness: 0.303
Normalized Mutual Information: 0.373

5、Ward hierarchical clustering
(A)获取数据集得到标签，使用Ward hierarchical clustering算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为,所得结果为
Homogeneity: 0.239
Normalized Mutual Information: 0.466
Completeness: 0.908

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用AgglomerativeClustering进行训练，部分参数使用affinity='euclidean', compute_full_tree='auto',connectivity=None,linkage='ward'，所得结果为：
Homogeneity: 0.458
Completeness: 0.303
Normalized Mutual Information: 0.373

7、DBSCAN
(A)获取数据集得到标签，使用DBSCAN算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为eps=3.5, min_samples=1.5,所得结果为：
Normalized Mutual Information: 0.498
Homogeneity: 0.519
Completeness: 0.477

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用DBSCAN进行训练，参数使用eps=0.05, min_samples=2所得结果为：
Homogeneity: 0.047
Completeness: 0.176
NMI: 0.090

8、Gaussian mixtures
(A)获取数据集得到标签，使用DBSCAN算法对数据集进行训练，使用评估函数比较训练前后得到的结果。训练时设置的参数为n_components=40，所得结果为：
Homogeneity: 0.863
Completeness: 0.584
Normalized Mutual Information:0.710

(B)将数据集中加载四个类别，分别为categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]，将数据集降维后，使用Gaussian mixtures进行训练，参数使用n_components=60所得结果为：
Homogeneity: 0.047
Completeness: 0.176
NMI: 0.090
