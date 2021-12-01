# A.I.D.D. 2021 Code Review

**[Artificial Intelligence Diabetes Datathon 2021](https://github.com/DatathonInfo/AIDD2021)**

Team: 영선없는영선팀
- **[Dongha Kim](https://github.com/kdha0727)**
- **[Dongjun Hwang](https://github.com/Druwa-Git)**
- **[Donggeon Bae](https://github.com/AttiBae)**
- **[Junho Lee](https://github.com/leejunho0421)**

---

# About Competition

- Objective: To predict Diabetes from 22 columns - `Binary Classification of Diabetes`
- How: Train and submit models on NSML (**[about NSML](NSML.md)**)

|Contest|Submission|Score|Rank|
|------|-----|-----|-----|
|Prelim|MLP|82.08|4 / 40|
|Final|MLP|82.22|15 / 20|

---

# About Dataset

- Tabular Dataset
- 22 Columns containing information related to diabetes
- Binary Labels (0, 1)
- Diabetes Diagnosis Criteria: (`FBG` >= 126 mg/dL) || (`HbA1c` >= 6.5 %)

## Columns

|Column|Description|
|---|---|
|`CDMID`|참가자 고유 아이디|
|`gender`|성별|
|`age`|나이|
|`date`|baseline 건강검진 일자|
|`Ht`|신장|
|`Wt`|체중|
|`BMI`|체질량지수 (BMI)|
|`SBP`|수축기 혈압|
|`DBP`|이완기 혈압|
|`PR`|맥박|
|`HbA1c`|**_당화혈색소_**|
|`FBG`|**_공복혈당_**|
|`TC`|총 콜레스테롤|
|`TG`|중성 지방|
|`LDL`|LDL 콜레스테롤|
|`HDL`|HDL 콜레스테롤|
|`Alb`|알부민|
|`BUN`|혈중요소질소|
|`Cr`|크레아티닌|
|`CrCl`|크레아티닌 청소율|
|`AST`|아스파테이트아미노 전이효소|
|`ALT`|알라닌아미노 전이효소|
|`GGT`|감마글루타밀 전이효소|
|`ALP`|알칼리 인산 분해효소|
|`date_E`|Endpoint 건강검진 일자|

## Data Distribution

|Label|Negative(0)|Positive(1)|
|---|---|---|
|Train|5581|515|
|Test|99|9|

- Label Distribution 만 봐도 알 수 있듯이 imbalance 가 너무 심했다.

---

# About Code
- Submissions
  - [`lightgbm.py`](lightgbm.py): lightgbm (Machine Learning)
  - [`logistic.py`](logistic.py): logistic regression via pytorch
  - [`mlp_base.py`](mlp_base.py): MLP - SOTA
  - [`mlp_variant.py`](mlp_variant.py): MLP with various channels and various activation
  - [`xbnet.py`](xbnet.py): Apply XBNet
  - [`tabnet.py`](tabnet.py): Apply TabNet
- Require
  - [`nsml`](nsml/__init__.py): Dummy Implementation for NSML Server-Side Package
  - [`setup.py`](setup.py): Environment Preparation
    - Includes Docker Image (Python + Pytorch) Information and Libraries

## Models

### 1. MLP
 
MLP with **[Perceptron Rule](https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde)**.

- add bias column
- construct first input layer to (input + output) * (2/3) = 16
- layers should be leq than (input + output) = 24
  - prevents overfitting
- each layer should be: `Linear -> BatchNorm -> Activation -> Dropout`

### 2. XBNet

See **[paper](https://arxiv.org/abs/2106.05239)** and **[implementation](https://github.com/tusharsarkar3/XBNet)**.

XGBoost 보다 뛰어난 Accuracy, 2021 에 발표된 모델로 diabetes dataset 에서 SOTA 달성.

### 3. TabNet

See **[Paper](https://arxiv.org/abs/1908.07442)** and **[Implementation](https://github.com/dreamquark-ai/tabnet)**.

CNN or MLP may not be the best fit for tabular data decision manifolds due to being vastly overparametrized – the lack of appropriate inductive bias often causes them to fail to find robust solutions for tabular decision manifolds.

TabNet selects features it use via attention mechanism, in each decision-making, resulting greater performance in tabular datasets.

### 4. ML Algorithms

## Preprocessing

### Scaling Continuous Variable

- Normalize
- MinMax
- Standard

딥러닝 모델들은 Training 과정에서 특별한 scaling 없이도 어느 정도 학습이 잘 되었다. (적용하지 않는 경우가 더 잘 되기도 했다.)

### Selection

- Lasso
- MI
- Fscore

이 역시 딥러닝 모델들에서 알아서 된다고 봐도 된다.

### Bias Column

- Adding 1 Bias Column

### Oversampling

- SMOTE (Best)
- ROSE
- [smote_variants](https://github.com/analyticalmindsltd/smote_variants)

Imbalanced Data 특성상 Oversampling 이 중요한 것 같다.

## Training

- Loss Function: `BCEWithLogitsLoss` 사용
- Optimizer: `Adam` 사용 (`AdamW` 사용도 고려해보라는 팀원분의 말씀이 있었으나 적용해보지 못함)
- LR Scheduler 로 `MultiStep`, [`CosineAnnealingWarmUpRestarts`](https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup/cosine_annealing_with_warmup.py) 사용
  - 오히려 column 적어 MultiStep 과 같은 Learning Rate 조절 방식이 결과가 잘 나왔다.

---

# Review

- 평가 기준을 애매하게 알려주었는데, inference pipeline 으로부터 충분히 Sensitivity 및 Specificity 의 중요성을 알 수 있었다.
  - Positive Data 작아서 Score 에 Positive Data 를 얼마나 잘 맞추느냐의 비중이 컸다. (Sensitivity)
- Imbalanced Data 특성상 Oversampling 이 중요한 것 같았다.
  - 오히려 데이터를 미세하게 변형하지 않고 양만 늘려주는 ROSE 성능이 더 높기도 했는데, 관련된 오버샘플링 방법들도 기회가 된다면 찾아보고 써봐야겠다.
- 다른 제출하신 분들 보니 Machine Learning 기반으로 많이 하신 것 같은데, 오히려 적은 Column 데이터는 Machine Learning 알고리즘이 더 나을 수도 있는 것 같다.
  - 여름부터 딥 러닝을 바로 시작하다 보니 Machine Learning 경험이 부족해서 이번 대회를 통해 ML 관련 방법론들을 여러 가지 알 수 있었다.
  - 시간 내서 Machine Learning 공부도 해보자

---
