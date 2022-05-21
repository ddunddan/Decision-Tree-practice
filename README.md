# Decision tree -practice

# 2 의사결정나무 실습



### 실험환경 구성

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer # 사용할 dataset (유방암데이터)
from sklearn.model_selection import train_test_split # 학습, 테스트set 구분
from sklearn.tree import export_graphviz # tree 시각화를 위해
# export_graphviz : 의사결정나무에 대한 graphviz dot data 생성하는 함수
import graphviz # tree 시각화
import sklearn.metrics as mt # 성능지표를 계산하기 위해 import
from sklearn.model_selection import cross_val_score, cross_validate # 교차검증

import warnings
warnings.filterwarnings('ignore')

import numpy as np

## 2-1 데이터 수집 및 전처리

shoppers = pd.read_csv('C:/Users/Soonchan Kwon/Desktop/online_shoppers_intention.csv')

shoppers 데이터는 인터넷 쇼핑몰 방문자 구매 의도 관련 데이터다. 

* "Administrative", "Administrative Duration", "Informational", "Informational Duration", Product Related","Product Related Duration"는<br>
* 방문자가 해당 세션에 방문한 페이지 수와 그 세션에서 보낸 시간을 의미한다.<br>
* "Bounce Rates"는 이탈률로 해당 들어온 페이지에서 나가는 방문자 비율이고 "Exit Rates"는 종료율로 방문자가 해당 페이지가 마지막 페이지 일 확률이며<br>
* "PageValues"는 거래를 완료하기 전에 방문한 웹 페이지의 평균 가치를 말하며 이들은 Google Analytics에서 측정할 수 있는 항목이다.<br>
* "SpecialDay"는 거래로 마무리될 가능성이 높은 특정 특별한 날(예: 어버이날, 발렌타인 데이)에 사이트 방문 시간이 얼마나 가까운지를 의미한다.
* "Revenue"는 Class rable로 해당 방문자가 거래를 했는지 여부를 알려준다. 

shoppers.head(10)

shoppers 데이터 셋 확인

### 데이터 전처리 1

shoppers.Month = shoppers.Month.replace('Feb', 0)
shoppers.Month = shoppers.Month.replace('Mar', 0)
shoppers.Month = shoppers.Month.replace('May', 0)
shoppers.Month = shoppers.Month.replace('June', 0)
shoppers.Month = shoppers.Month.replace('Jul', 0)
shoppers.Month = shoppers.Month.replace('Aug', 1)
shoppers.Month = shoppers.Month.replace('Sep', 1)
shoppers.Month = shoppers.Month.replace('Oct', 1)
shoppers.Month = shoppers.Month.replace('Nov', 1)
shoppers.Month = shoppers.Month.replace('Dec', 1)

"Month"에 해당하는 값을 수치형으로 변환시켜주었다.<br>
다만 월별로 크기적 의미를 가지고 있진 않고 그렇다고 각각 더미 변수화 시키기엔<br>
종류가 너무 많아질 것이라 의식해 계절적 특징을 가지는 상반기, 하반기로만 분류 변환 하였다. (5개씩)

shoppers.VisitorType = shoppers.VisitorType.replace('Returning_Visitor', 0)
shoppers.VisitorType = shoppers.VisitorType.replace('New_Visitor', 1)
shoppers.VisitorType = shoppers.VisitorType.replace('Other', 2)



* "VisitorType" 또한 수치형으로 변경시켜주었다.

shoppers["Weekend"] = shoppers["Weekend"].astype(int)

shoppers["Revenue"] = shoppers["Revenue"].astype(int)

* "Weekend"와 "Revenue" 또한 bool값에서 수치형으로 변환시켰다. <br>

<br>
### 데이터 전처리 2 
<br>

import seaborn as sns
import matplotlib.pyplot as plt
shoppers_res = plt.boxplot(shoppers[['ProductRelated','PageValues']])
plt.xticks([1,2], ['ProductRelated','PageValues'])
plt.show()

이상치가 존재할만한 Feature 값을 boxplot 함수를 통해 확인해주었다.<br>
Feature 특성상 이상치가 다수 존재할 수 밖에 없기에 따로 이상치를 생성하지 않고 그대로 두었다. <br>

x = np.array(pd.DataFrame(shoppers, columns=['Administrative','Administrative_Duration',
                                             'Informational', 'Informational_Duration'
                                            ,'ProductRelated', 'ProductRelated_Duration'
                                            , 'BounceRates', 'ExitRates', 'PageValues'
                                            , 'SpecialDay', 'Month', 'OperatingSystems'
                                            , 'Browser', 'Region','TrafficType','VisitorType','Weekend']))

y = np.array(pd.DataFrame(shoppers, columns=['Revenue']))

shoppers는 Pandas의 df형태의 데이터 셋이기에 <br>
학습데이터를 분리하기 위한 train_test_split 함수에 적용하기 편하게 x와, y를 numpy array 형태로 정의해주었다. <br>

seed = 777

# 학습, 테스트 데이터 분리 (0.7:0.3)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.7, random_state=seed)

train_test_split 함수를 통해 학습 데이터와 테스트 데이터를 7:3 비율로 분리하였다. <br>

x_train

x_test

y_train

y_test

학습데이터 분리를 완료하였다. <br>

<br>

## 2-2 의사결정나무 구축 및 실험비교

<br>

dt_clf = DecisionTreeClassifier(random_state=777)
dt_clf.fit(x_train, y_train) # 학습

데이터를 포함한 의사결정 나무를 구축하였다. <br>

y_pred = dt_clf.predict(x_test) 

# 학습결과 평가
print("Train_Accuracy : ", dt_clf.score(x_train, y_train), '\n')
print("Test_Accuracy : ", dt_clf.score(x_test, y_test), '\n')

accuracy = mt.accuracy_score(y_test, y_pred)
recall = mt.recall_score(y_test, y_pred)
precision = mt.precision_score(y_test, y_pred)
f1_score = mt.f1_score(y_test, y_pred)
matrix = mt.confusion_matrix(y_test, y_pred)

print('Accuracy: ', format(accuracy,'.2f'),'\n')
print('Recall: ', format(recall,'.2f'),'\n')
print('Precision: ', format(precision,'.2f'),'\n')
print('F1_score: ', format(f1_score,'.2f'),'\n')
print('Confusion Matrix:','\n', matrix)

#### 의사결정나무 모델의 학습 결과를 평가하였다.
<br>
* Train_Accuracy는 1.0으로 학습 데이터에 완벽하게 과적합 되어있음을 알 수 있다. <br>

* Test_Accuracy는 0.85로 보여진다. <br>

* Recall 값은 0.54 , Precision:  0.53 이며  F1_score 0.54 임을 알 수 있다.<br>

<br>

### Cross Validation

<br>

# 교차검증

# 각 폴드의 스코어 
scores = cross_val_score(dt_clf, x, y, cv = 5)

print('Averaged results of cross validation: ', scores.mean())

pd.DataFrame(cross_validate(dt_clf, x, y, cv =5))

교차검증 후 test set에 대한 스코어(정확도)를 알아보았다.

dt_clf.score(x_test, y_test)

<br>

### 가지치기 후 모델 구축

<br>

pruned_dt_clf = DecisionTreeClassifier(max_depth=3, random_state=156) # max_depth=4으로 제한
pruned_dt_clf .fit(x_train, y_train)


max_depth 를 5로 가지치기를 진행한 후 새로운 의사결정 나무 모델을 구축하였다.

scores = cross_val_score(pruned_dt_clf, x, y, cv = 5)

print('Averaged results of cross validation: ', scores.mean())

# 학습결과 평가
print("Train_Accuracy : ", pruned_dt_clf.score(x_train, y_train), '\n')
print("Test_Accuracy : ", pruned_dt_clf.score(x_test, y_test), '\n')

학습 결과 평가를 서로 비교했을 때, <br>

* 가지치기 전 모델은 <br>

Test_Accuracy :  0.8500753099293246 <br>

* 가지치기 후 모델은 <br>

Test_Accuracy :  0.8833275402618468 <br>

로 가지치기 후 test set에 대한 정확도가 향상됨을 확인할 수 있다. 

<br>

## 2.3 의사결정나무 학습모델 가시화 및 설명

<br>

shoppers_feature = shoppers.drop(['Revenue'],axis=1)
export_graphviz(dt_clf, out_file="tree.dot", class_names = ["Not Purchase", "Purchase"], feature_names = shoppers_feature.columns, impurity=True, filled=True) 

feture_names 값에 할당하기위해 lable feature인 'Revenue를 제거한 후 <br>
export_graphviz( )의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성하였다.

print('[ max_depth의 제약이 없는 경우의 Decision Tree 시각화 ]')
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

<br> <br>

max_depth의 제약이 없는 경우 엄청난 크기의 Decision Tree가 형성되었음을 알 수 있다.

export_graphviz(pruned_dt_clf, out_file="prunedtree.dot", class_names = ["Not Purchase", "Purchase"], feature_names = shoppers_feature.columns, impurity=True, filled=True)

<br>
그렇다면 가지치기 후의 트리를 시각화 해보았다.

print('[ max_depth가 3인 경우의 Decision Tree 시각화 ]')
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("prunedtree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

<br> max_depth를 제약하고 나니 Decision Tree가 좀 더 단순하게 잘 보인다.
<br>

* PageValues를 기준으로 먼저 분류가 되고 <br>
* 여러 feature들의 분류 기준을 통하여 분류한 구매 여부를 확인할 수 있다. <br>
* 따라서 PageValues가 가장 중요한 기준 feature라고 할 수 있다.

<br>

### 중요도 확인 

<br>

feature_importance_values = pruned_dt_clf.feature_importances_
# Top 중요도로 정렬하고, 쉽게 시각화하기 위해 Series 변환
feature_importances = pd.Series(feature_importance_values, index=shoppers_feature.columns)
# 중요도값 순으로 Series를 정렬
feature_top5 = feature_importances.sort_values(ascending=False)[:5] # 10개 혹은 20개 등 개수를 바꾸고 싶다면 이 부분을 변경


plt.figure(figsize=[8, 6])
plt.title('Feature Importances Top 5')
sns.barplot(x=feature_top5, y=feature_top5.index)
plt.show()


의사결정나무 시각화를 통해 알아보았지만 어떤 feature 값이 분류에 더 큰 영향을 미치는지 <br> 알아보기 위해 feature 중요도를 시각화 해보았다. <br>

* tree 시각화 값에서 알 수 있듯이 PageVaules의 중요도가 매우 높음을 알 수 있다. <br>
* 그 밖에 Administrative, Month 등이 높은 중요도를 뒤따르지만 PageValues값에는 못미친다.

<br>

## 2.4 의사결정나무의 특징 코멘트 

<br>

### 의사결정나무의 특징 

<br>



* 이번 분석에서 알 수 있듯이 의사결정나무는 설명변수로 목표변수를 분류나 예측하는데에 있어서 쓰이는 분석 방법이다. <br>
* 말 그대로 어떠한 행동을 하는데에 있어 의사를 결정하는 방법을 확률에 의해 알려주는 것이다. <br>
* 방금 알아 본 것처럼 어떠한 feature 값이 중요한 영향을 미치는지 확인할 수 있고, 어떠한 기준에 의해 분류했는지, 예측했는지 기준을 명확하게 알 수 있다. 

<br>

### 의사 결정나무의 장점
<br>

여러가지 장점이 있겠지만 분석을 통해 직접 체험해 본 장점은 다음과 같다. <br>

* 데이터의 이상치와 결측치에 자유롭다. <br>
    결측치는 포함하진 않았지만 이상치가 다수 포함되어 있음에도 분석에 문제가 없었고 분석 알고리즘을 보았을 때, 결측치 값이 존재하더라도 문제가 없을 것 같다.
  <br>
  
* 결과 해석에 매우 용이하다. <br>
    시각화한 트리는 분석자로 하여금 정말 직관적인 해석을 가능하게 한다. 어떤 변수로 어떤 기준을 통해 의사를 결정할 수 있는지 그 제시가 명확하다는 점이 해석을 용이하게 한다. <br>
    
* 가정에 자유롭다 <br>
    이전 회귀모델로 인한 실습을 했을 때엔 정규성, 독립성, 등분산성 등의 가정들을 만족시켜줘야했다. 하지만 의사결정나무를 사용했을 시 그 가정에 자유롭다. <br>

### 의사 결정나무의 단점 <br>

뚜렷한 장점이 이렇게 존재하지만 그럼에도 내가 느낀 단점은 다음과 같다. <br>

* 학습데이터 과대적합 문제가 발생한다. <br>
    위의 학습 모델 정확도를 통해 확인 했듯이 학습데이터에 의한 과대적합 문제가 발생한다. 따라서 자신의 데이터에 맞는 적정한 가지치기를 방법을 통해 과적합 문제를 해결해야 한다. <br>
    
* 변수가 연속형일 경우 알맞지 않다. <br>
    분리 기준에 의해 가지를 나누어갈 때 feature가 연속형일 경우 구간화 처리 후 판단하기에 적당하지 않다. <br>
    
* 데이터의 변화에 민감하다. <br>
    실제로 분류 중간에 feature 변수를 지우기도 해보고 데이터 값을 실수로 누락도 시켰었는데 그 때마다 모델이 상대적으로 안정적이지 못한 모습을 보여줬다. 
    
* 복잡한 대상에 대한 해석이 어렵다. <br>
    내가 구축한 모형에 대한 해석을 가지고 실제 Online 쇼핑몰 마케팅 도구로 활용한다고 했을 때, 어떠한 페이지를 중점적으로, 어떠한 전략을 실시해야할지 전체 모형으로는 쉽게 의견을 제시하기 어렵다고 생각한다. 특히 분류가 아닌 예측일 때는 더 심해질 것이라고 생각된다.
    
