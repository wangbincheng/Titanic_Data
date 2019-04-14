import pandas as pd
# 数据加载
train_data = pd.read_csv('./Titanic_Data/train.csv')
test_data = pd.read_csv('./Titanic_Data/test.csv')
# 数据探索
print(train_data.info())
print('-'*30)
print(test_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
print('-'*30)
print(test_data.describe())
print('-'*30)
print(test_data.describe(include=['O']))
print('-'*30)
print(test_data.head())
print('-'*30)
print(test_data.tail())

# 模块2：数据清洗

#使用平均年龄来填充年龄中的NaN值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
#使用票价的均值填充票价的NaN值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

# 观察下 Embarked 字段的取值
print(train_data['Embarked'].value_counts())

#使用登陆最多的港口来填充登陆港口的NaN值
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 模块 3：特征选择
# 特征选择
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features=train_data[features]
train_labels=train_data['Survived']
test_features=test_data[features]

# 将特征值的字符串转成数值类型
from sklearn.feature_extraction import DictVectorizer
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 模块 4：决策树模型
from sklearn.tree import DecisionTreeClassifier
# 构造ID3决策树
clf=DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features,train_labels)

# 模块 5：模型预测 & 评估
test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

#得到决策树准确率
acc_decision_tree=round(clf.score(train_features,train_labels),6)
print(u'score 准确率为 %.4lf'%acc_decision_tree)

import numpy as np
from sklearn.model_selection import cross_val_score
# 使用 K 折交叉验证 统计决策树准确率
print(u'cross_val_score 准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

#模块6: 决策树可视化
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Titanic.pdf")
