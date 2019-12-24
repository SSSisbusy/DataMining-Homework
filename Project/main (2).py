#!/usr/bin/env python
# coding: utf-8

import json
import random
from pyjarowinkler import distance
import numpy as np

with open("cna_data/train_author.json", "r") as f2:
    author_data = json.load(f2)

with open("cna_data/train_pub.json", "r") as f2:
    pubs_dict = json.load(f2)
# 论文id title author.name等
print(len(author_data))

name_train = set()

for name in author_data:
    persons = author_data[name]
    if (len(persons) > 10):
        name_train.add((name))

print(len(name_train))


paper2aid2name = {}

for author_name in name_train:
    persons = author_data[author_name]
    for person in persons:
        paper_list = persons[person]
        for paper_id in paper_list:
            paper2aid2name[paper_id] = (author_name, person)

print(len(paper2aid2name))
print(paper2aid2name)

total_paper_list = list(paper2aid2name.keys())

# 采样10000篇paper作为训练集
train_paper_list = random.sample(total_paper_list, 3000)

# 把采样的500篇paper转变成对应的训练例子，一个训练例子包含paper和正例作者以及5个负例作者（正负例比=1：5）
train_instances = []
for paper_id in train_paper_list:

    # 保存对应的正负例
    pos_ins = set()
    neg_ins = set()

    paper_author_name = paper2aid2name[paper_id][0]
    paper_author_id = paper2aid2name[paper_id][1]

    pos_ins.add((paper_id, paper_author_id))

    # 获取同名的所有作者(除了本身)作为负例的candidate
    persons = list(author_data[paper_author_name].keys())
    persons.remove(paper_author_id)
    assert len(persons) == (len(list(author_data[paper_author_name].keys())) - 1)

    # 每个正例采样5个负例
    neg_author_list = random.sample(persons, 5)
    for i in neg_author_list:
        neg_ins.add((paper_id, i))

    train_instances.append((pos_ins, neg_ins))

print(len(train_instances))
from pyjarowinkler import distance


# 对author_name 进行清洗
def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    # x = [k.strip() for k in name.lower().strip().replace("-", "").replace("_", ' ').split()]
    full_name = ' '.join(x)
    name_part = full_name.split()
    if (len(name_part) >= 1):
        return full_name
    else:
        return None


# 找出paper中author_name所对应的位置
def delete_main_name(author_list, name):
    score_list = []
    name = clean_name(name)
    author_list_lower = []
    for author in author_list:
        author_list_lower.append(author.lower())
    name_split = name.split()
    for author in author_list_lower:
        # lower_name = author.lower()
        score = distance.get_jaro_distance(name, author, winkler=True, scaling=0.3)
        author_split = author.split()
        inter = set(name_split) & set(author_split)
        alls = set(name_split) | set(author_split)
        score += round(len(inter) / len(alls), 6)
        score_list.append(score)

    rank = np.argsort(-np.array(score_list))
    return_list = [author_list_lower[i] for i in rank[1:]]

    return return_list, rank[0]


# 训练集特征生成函数
def process_feature(pos_ins, paper_coauthors):
    feature_list = []

    paper = pos_ins[0]
    author = pos_ins[1]

    paper_name = paper2aid2name[paper][0]

    # 从作者的论文列表中把该篇论文去掉，防止训练出现bias
    doc_list = []
    for doc in author_data[paper_name][author]:
        if (doc != paper):
            doc_list.append(doc)
    for doc in doc_list:
        if doc == paper:
            print("error!")
            exit()

    # 保存作者的所有paper的coauthors以及各自出现的次数(作者所拥有论文的coauthors)
    candidate_authors_int = defaultdict(int)

    total_author_count = 0
    for doc in doc_list:

        doc_dict = pubs_dict[doc]
        author_list = []

        paper_authors = doc_dict['authors']
        paper_authors_len = len(paper_authors)
        paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

        for author in paper_authors:
            clean_author = clean_name(author['name'])
            if (clean_author != None):
                author_list.append(clean_author)
        if (len(author_list) > 0):
            # 获取paper中main author_name所对应的位置
            _, author_index = delete_main_name(author_list, paper_name)

            # 获取除了main author_name外的coauthor
            for index in range(len(author_list)):
                if (index == author_index):
                    continue
                else:
                    candidate_authors_int[author_list[index]] += 1
                    total_author_count += 1

    # author 的所有不同coauthor name
    author_keys = list(candidate_authors_int.keys())

    if ((len(author_keys) == 0) or (len(paper_coauthors) == 0)):
        feature_list.extend([0.] * 5)
    else:
        co_coauthors = set(paper_coauthors) & set(author_keys)
        coauthor_len = len(co_coauthors)

        co_coauthors_ratio_for_paper = round(coauthor_len / len(paper_coauthors), 6)
        co_coauthors_ratio_for_author = round(coauthor_len / len(author_keys), 6)

        coauthor_count = 0
        for coauthor_name in co_coauthors:
            coauthor_count += candidate_authors_int[coauthor_name]

        co_coauthors_ratio_for_author_count = round(coauthor_count / total_author_count, 6)

        feature_list.extend([coauthor_len, co_coauthors_ratio_for_paper, co_coauthors_ratio_for_author, coauthor_count,
                             co_coauthors_ratio_for_author_count])

    #         print(feature_list)
    return feature_list

from collections import defaultdict

pos_features = []
neg_features = []

print(len(train_instances))

for ins in train_instances:

    pos_set = ins[0]
    neg_set = ins[1]
    paper_id = list(pos_set)[0][0]
    paper_name = paper2aid2name[paper_id][0]

    author_list = []
    # 获取paper的coauthors
    paper_coauthors = []

    paper_authors = pubs_dict[paper_id]['authors']
    paper_authors_len = len(paper_authors)
    # 只取前50个author以保证效率
    paper_authors = random.sample(paper_authors, min(200, paper_authors_len))

    for author in paper_authors:
        clean_author = clean_name(author['name'])
        if (clean_author != None):
            author_list.append(clean_author)
    if (len(author_list) > 0):
        # 获取paper中main author_name所对应的位置
        _, author_index = delete_main_name(author_list, paper_name)

        # 获取除了main author_name外的coauthor
        for index in range(len(author_list)):
            if (index == author_index):
                continue
            else:
                paper_coauthors.append(author_list[index])

        for pos_ins in pos_set:
            pos_features.append(process_feature(pos_ins, paper_coauthors))

        for neg_ins in neg_set:
            neg_features.append(process_feature(neg_ins, paper_coauthors))

print(np.array(pos_features).shape)
print(np.array(neg_features).shape)

from sklearn.svm import SVC
from sklearn.externals import joblib
from math import log
import operator

# 构建svm正负例
svm_train_ins = []
for ins in pos_features:
    svm_train_ins.append((ins, 1))

for ins in neg_features:
    svm_train_ins.append((ins, 0))

print(np.array(svm_train_ins).shape)

random.shuffle(svm_train_ins)

x_train = []
y_train = []
for ins in svm_train_ins:
    x_train.append(ins[0])
    y_train.append(ins[1])

clf = SVC(probability=True, C=0.5, kernel='linear', decision_function_shape='ovo')
clf.fit(x_train, y_train)


# 训练集中的作者论文信息
with open("cna_data/whole_author_profile.json", "r") as f2:
    test_author_data = json.load(f2)

# 训练集的论文元信息
with open("cna_data/whole_author_profile_pub.json", "r") as f2:
    test_pubs_dict = json.load(f2)

# 待分配论文集
with open("cna_test_data/cna_test_unass_competition.json", "r") as f2:
    unass_papers = json.load(f2)

with open("cna_test_data/cna_test_pub.json", "r") as f2:
    unass_papers_dict = json.load(f2)

# with open("cna_data/new_test_author_data.json", 'r') as files:
#     new_test_author_data = json.load(files)
# 简单处理whole_author_profile，将同名的作者合并：
# 为了效率，预处理new_test_author_data中的paper，将其全部处理成paper_id + '-' + author_index的形式。
new_test_author_data = {}
for author_id, author_info in test_author_data.items():
    author_name = author_info['name']
    author_papers = author_info['papers']
    newly_papers = []

    for paper_id in author_papers:

        paper_authors = test_pubs_dict[paper_id]['authors']
        paper_authors_len = len(paper_authors)

        # 只利用author数小于50的paper，以保证效率
        if (paper_authors_len > 150):
            continue
        #         paper_authors = random.sample(paper_authors, min(50, paper_authors_len))
        author_list = []
        for author in paper_authors:
            clean_author = clean_name(author['name'])
            if (clean_author != None):
                author_list.append(clean_author)
        if (len(author_list) > 0):
            # 获取paper中main author_name所对应的位置
            _, author_index = delete_main_name(author_list, paper_name)

            new_paper_id = str(paper_id) + '-' + str(author_index)
            newly_papers.append(new_paper_id)

    if (new_test_author_data.get(author_name) != None):
        new_test_author_data[author_name][author_id] = newly_papers
    else:
        tmp = {}
        tmp[author_id] = newly_papers
        new_test_author_data[author_name] = tmp
print(len(new_test_author_data))


# test集的特征生成函数，与train类似
def process_test_feature(pair, new_test_author_data, test_pubs_dict, paper_coauthors):
    feature_list = []

    paper = pair[0]
    author = pair[1]
    paper_name = pair[2]

    doc_list = new_test_author_data[paper_name][author]

    # 保存作者的所有coauthors以及各自出现的次数(作者所拥有论文的coauthors)
    candidate_authors_int = defaultdict(int)

    total_author_count = 0
    for doc in doc_list:
        doc_id = doc.split('-')[0]
        author_index = doc.split('-')[1]
        doc_dict = test_pubs_dict[doc_id]
        author_list = []

        paper_authors = doc_dict['authors']
        paper_authors_len = len(paper_authors)
        paper_authors = random.sample(paper_authors, min(100, paper_authors_len))

        for author in paper_authors:
            clean_author = clean_name(author['name'])
            if (clean_author != None):
                author_list.append(clean_author)
        if (len(author_list) > 0):

            # 获取除了main author_name外的coauthor
            for index in range(len(author_list)):
                if (index == author_index):
                    continue
                else:
                    candidate_authors_int[author_list[index]] += 1
                    total_author_count += 1

    author_keys = list(candidate_authors_int.keys())

    if ((len(author_keys) == 0) or (len(paper_coauthors) == 0)):
        feature_list.extend([0.] * 5)
    else:
        co_coauthors = set(paper_coauthors) & set(author_keys)
        coauthor_len = len(co_coauthors)

        co_coauthors_ratio_for_paper = round(coauthor_len / len(paper_coauthors), 6)
        co_coauthors_ratio_for_author = round(coauthor_len / len(author_keys), 6)

        coauthor_count = 0
        for coauthor_name in co_coauthors:
            coauthor_count += candidate_authors_int[coauthor_name]

        co_coauthors_ratio_for_author_count = round(coauthor_count / total_author_count, 6)

        feature_list.extend([coauthor_len, co_coauthors_ratio_for_paper, co_coauthors_ratio_for_author, coauthor_count,
                             co_coauthors_ratio_for_author_count])

    #         print(feature_list)
    return feature_list


# In[66]:


print(len(unass_papers))

count = 0

# 存储paper的所有candidate author id
paper2candidates = defaultdict(list)
# 存储对应的paper与candidate author的生成特征
paper2features = defaultdict(list)

for u_p in u4nass_papers:
    paper_id = u_p.split('-')[0]
    author_index = int(u_p.split('-')[1])
    author_list = []

    # 获取paper的coauthors
    paper_coauthors = []
    paper_name = ''
    paper_authors = unass_papers_dict[paper_id]['authors']
    #     paper_authors_len = len(paper_authors)
    #     paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

    for author in paper_authors:
        clean_author = clean_name(author['name'])
        if (clean_author != None):
            author_list.append(clean_author)
    if (len(author_list) > 0):

        # 获取除了main author_name外的coauthor
        for index in range(len(author_list)):
            if (index == author_index):
                continue
            else:
                paper_coauthors.append(author_list[index])

    # 简单使用精确匹配找出candidate_author_list
    if(clean_name(paper_authors[author_index]['name']) == None):
        continue;
    paper_name = '_'.join(clean_name(paper_authors[author_index]['name']).split())
    if (new_test_author_data.get(paper_name) != None):
        candidate_author_list = new_test_author_data[paper_name]
        for candidate_author in candidate_author_list:
            pair = (paper_id, candidate_author, paper_name)
            paper2candidates[paper_id].append(candidate_author)
            paper2features[paper_id].append(
                process_test_feature(pair, new_test_author_data, test_pubs_dict, paper_coauthors))
        count += 1
print(count)
assert len(paper2candidates) == len(paper2features)
print(len(paper2candidates))

result_dict = defaultdict(list)
for paper_id, ins_feature_list in paper2features.items():
    score_list = []
    for ins in ins_feature_list:
        # 利用svm对一篇paper的所有candidate author去打分，利用分数进行排序，取top-1 author作为预测的author
        prob_pred = clf.predict_proba([ins])[:, 1]
        score_list.append(prob_pred[0])
    rank = np.argsort(-np.array(score_list))
    # 取top-1 author作为预测的author
    predict_author = paper2candidates[paper_id][rank[0]]
    result_dict[predict_author].append(paper_id)

with open("cna_data/result.json", 'w') as files:
    json.dump(result_dict, files, indent=4)

