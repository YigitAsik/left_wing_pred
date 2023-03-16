from data.YigasHelpers import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

gss = pd.read_csv('data/gss_years/2013/gss_data.csv')

df = gss.loc[gss["year"] >= 2008, ["caseid", "year", "age", "sex", "race", "degree", "marital", "childs",
                 "polviews", "natenvir", "confed", "conarmy", "conjudge", "class",
                 "abany", "helppoor", "helpsick", "helpblk"]]

df.columns = [col.upper() for col in df.columns]

df = df.loc[df["POLVIEWS"].notna(), :]

polviews_encode = {
    'Extremely Liberal':'1',
    'Liberal':'2',
    'Slightly Liberal': '3',
    'Moderate': '4',
    'Slightly Conservative': '5',
    'Conservative': '6',
    'Extrmly Conservative': '7'
}
df['POLVIEWS'] = df['POLVIEWS'].replace(polviews_encode)
df["POLVIEWS"] = df["POLVIEWS"].astype(int)
df.drop(df.loc[df["POLVIEWS"] == 4, :].index, inplace=True)

df.POLVIEWS.unique()

train, rem = train_test_split(df, train_size=.7, stratify=df.POLVIEWS, random_state=42)
valid, test = train_test_split(rem, test_size=.5, stratify=rem.POLVIEWS, random_state=42)


train.head()

# codebookta value-label ikilisinden görüleceği üzere "Don't know" diyenleri NA kodlamışlar.
# o sebeple DK diye kodluyorum null değerleri.
##  CONFIDENCE IN INSTITUTIONS
conf_govt = [col for col in train.columns if "CON" in col]
for col in conf_govt:
    train[col].fillna("DK", inplace=True)
    valid[col].fillna("DK", inplace=True)
    test[col].fillna("DK", inplace=True)


## NATIONAL PROBLEMS - environment
nat = [col for col in train.columns if "NAT" in col]
for col in nat:
    train[col].fillna("DK", inplace=True)
    valid[col].fillna("DK", inplace=True)
    test[col].fillna("DK", inplace=True)


## ABORTION
abor = [col for col in train.columns if 'AB' in col and col not in conf_govt and nat]
for col in abor:
    train[col].fillna('DK', inplace=True)
    valid[col].fillna("DK", inplace=True)
    test[col].fillna('DK', inplace=True)

## SHOULD GOVT HELP
hlp = [col for col in train.columns if 'HELP' in col]
for col in hlp:
    train[col].fillna('DK', inplace=True)
    valid[col].fillna("DK", inplace=True)
    test[col].fillna('DK', inplace=True)

missing_values_table(train)

train.dropna(inplace=True)
valid.dropna(inplace=True)
test.dropna(inplace=True)

train["CHILDS"] = train["CHILDS"].astype(int)
valid["CHILDS"] = valid["CHILDS"].astype(int)
test["CHILDS"] = test["CHILDS"].astype(int)

train["AGE"] = train["AGE"].astype(int)
valid["AGE"] = valid["AGE"].astype(int)
test["AGE"] = test["AGE"].astype(int)

train.loc[train["RACE"] != "White", "RACE"] = "Other"
valid.loc[valid["RACE"] != "White", "RACE"] = "Other"
test.loc[test["RACE"] != "White", "RACE"] = "Other"

train['IS_LEFT'] = train['POLVIEWS']
valid['IS_LEFT'] = valid['POLVIEWS']
test['IS_LEFT'] = test['POLVIEWS']

train.loc[train['POLVIEWS'] < 4, 'IS_LEFT'] = '1'
train.loc[train['POLVIEWS'] > 4, 'IS_LEFT'] = '0'
valid.loc[valid['POLVIEWS'] < 4, 'IS_LEFT'] = '1'
valid.loc[valid['POLVIEWS'] > 4, 'IS_LEFT'] = '0'
test.loc[test['POLVIEWS'] < 4, 'IS_LEFT'] = '1'
test.loc[test['POLVIEWS'] > 4, 'IS_LEFT'] = '0'

train['IS_LEFT'] = train['IS_LEFT'].astype(int)
valid['IS_LEFT'] = valid['IS_LEFT'].astype(int)
test['IS_LEFT'] = test['IS_LEFT'].astype(int)

test.IS_LEFT.unique()

cat_cols, num_cols, cat_but_car = grab_col_names(train)

for col in num_cols:
    target_summary_with_num(train, 'POLVIEWS', col)


fig = plt.figure(figsize=(9,8))
g = sns.barplot(data=train,
                x="POLVIEWS", y="AGE",
                palette="viridis", errorbar=('ci', False))
show_values(g)
plt.show()

for col in cat_cols:
    if col not in ['POLVIEWS', 'IS_LEFT']:
        print('Col name:' + str(col))
        target_summary_with_cat(train, 'POLVIEWS', col)
    else:
        continue

for col in cat_cols:
    if col not in ['POLVIEWS', 'IS_LEFT']:
        cc = train.groupby(str(col)).agg({'POLVIEWS': 'mean'})
        cc.reset_index(inplace=True)
        fig = plt.figure(figsize=(9, 8))

        g = sns.barplot(data=cc,
                        x=str(col), y="POLVIEWS",
                        palette="viridis")
        show_values(g)

        g.set_title(str(col))
        g.set_xlabel("ANSWERS")

        g.yaxis.set_minor_locator(AutoMinorLocator(5))

        g.tick_params(which="both", width=2)
        g.tick_params(which="major", length=6)
        g.tick_params(which="minor", length=4)
        fig.show()
    else:
        pass

# Bu noktada hipotez testi yaparak da istatistiksel olarak anlamlı bir fark var mı
# diye bakmak mümkün ancak literatürde bilinenlerden dolayı böyle bir yol izledim.
# Kezâ keşifsel olarak bakınca da (plotlardan görüldüğü üzere)
# bunun anlamlı olduğunu destekler nitelikte data.

for col in train.columns:
    if "HELP" in col:
        train.loc[train[col] == "DK", col] = "Agree With Both"
    elif "CON" in col:
        train.loc[train[col] == "DK", col] = "Only Some"
    elif "ENVIR" in col:
        train.loc[train[col] == "DK", col] = "About Right"
    else:
        continue

for col in valid.columns:
    if "HELP" in col:
        valid.loc[valid[col] == "DK", col] = "Agree With Both"
    elif "CON" in col:
        valid.loc[valid[col] == "DK", col] = "Only Some"
    elif "ENVIR" in col:
        valid.loc[valid[col] == "DK", col] = "About Right"
    else:
        continue

for col in test.columns:
    if "HELP" in col:
        test.loc[test[col] == "DK", col] = "Agree With Both"
    elif "CON" in col:
        test.loc[test[col] == "DK", col] = "Only Some"
    elif "ENVIR" in col:
        test.loc[test[col] == "DK", col] = "About Right"
    else:
        continue

for col in num_cols:
    target_summary_with_num(train, 'POLVIEWS', col)

for col in cat_cols:
    print('Col name:' + str(col))
    target_summary_with_cat(train, 'POLVIEWS', col)

for col in cat_cols:
    if col not in ['POLVIEWS', 'IS_LEFT']:
        cc = train.groupby(str(col)).agg({'POLVIEWS': 'mean'})
        cc.reset_index(inplace=True)
        fig = plt.figure(figsize=(9, 8))

        g = sns.barplot(data=cc,
                        x=str(col), y="POLVIEWS",
                        palette="viridis")
        show_values(g)

        g.set_title(str(col))
        g.set_xlabel("ANSWERS")

        g.yaxis.set_minor_locator(AutoMinorLocator(5))

        g.tick_params(which="both", width=2)
        g.tick_params(which="major", length=6)
        g.tick_params(which="minor", length=4)
        fig.show()
    else:
        pass


## LIB_SCORE

# train
train['HELPPOOR_FLAG'] = train['HELPPOOR']
hlppr_ec = {
    'Agree With Both': '2',
    'Govt Action': '3',
    'People Help Selves': '1'
}
train['HELPPOOR_FLAG'] = train['HELPPOOR_FLAG'].replace(hlppr_ec)
train["HELPPOOR_FLAG"] = train["HELPPOOR_FLAG"].astype(int)


train["HELPSICK_FLAG"] = train["HELPSICK"]
hlpsck_ec = {
    'Agree With Both': '2',
    'Govt Should Help': '3',
    'People Help Selves': '1'
}
train["HELPSICK_FLAG"] = train["HELPSICK_FLAG"].replace(hlpsck_ec)
train["HELPSICK_FLAG"] = train["HELPSICK_FLAG"].astype(int)


train["HELPBLK_FLAG"] = train["HELPBLK"]
hlpblk_ec = {
    'Agree With Both': '2',
    'Govt Help Blks': '3',
    'No Special Treatment': '1'
}
train["HELPBLK_FLAG"] = train["HELPBLK_FLAG"].replace(hlpblk_ec)
train["HELPBLK_FLAG"] = train["HELPBLK_FLAG"].astype(int)


train["ABANY_FLAG"] = train["ABANY"]
abany_ec = {
    'DK':'2',
    'Yes':'3',
    'No':'1'
}
train["ABANY_FLAG"] = train["ABANY_FLAG"].replace(abany_ec)
train["ABANY_FLAG"] = train["ABANY_FLAG"].astype(int)

train["NATENVIR_FLAG"] = train["NATENVIR"]
natenv_ec = {
    'About Right': '2',
    'Too Much': '3',
    'Too Little': '1'
}
train["NATENVIR_FLAG"] = train["NATENVIR_FLAG"].replace(natenv_ec)
train["NATENVIR_FLAG"] = train["NATENVIR_FLAG"].astype(int)


train["LIB_SCORE"] = train["ABANY_FLAG"] + train["NATENVIR_FLAG"] + train["HELPBLK_FLAG"] + train["HELPSICK_FLAG"] + train["HELPPOOR_FLAG"]

# valid
valid['HELPPOOR_FLAG'] = valid['HELPPOOR']
valid["HELPPOOR_FLAG"] = valid["HELPPOOR_FLAG"].replace(hlppr_ec)
valid["HELPPOOR_FLAG"] = valid["HELPPOOR_FLAG"].astype(int)


valid["HELPSICK_FLAG"] = valid["HELPSICK"]
valid["HELPSICK_FLAG"] = valid["HELPSICK_FLAG"].replace(hlpsck_ec)
valid["HELPSICK_FLAG"] = valid["HELPSICK_FLAG"].astype(int)


valid["HELPBLK_FLAG"] = valid["HELPBLK"]
valid["HELPBLK_FLAG"] = valid["HELPBLK_FLAG"].replace(hlpblk_ec)
valid["HELPBLK_FLAG"] = valid["HELPBLK_FLAG"].astype(int)


valid["ABANY_FLAG"] = valid["ABANY"]
valid["ABANY_FLAG"] = valid["ABANY_FLAG"].replace(abany_ec)
valid["ABANY_FLAG"] = valid["ABANY_FLAG"].astype(int)


valid["NATENVIR_FLAG"] = valid["NATENVIR"]
valid["NATENVIR_FLAG"] = valid["NATENVIR_FLAG"].replace(natenv_ec)
valid["NATENVIR_FLAG"] = valid["NATENVIR_FLAG"].astype(int)


valid["LIB_SCORE"] = valid["ABANY_FLAG"] + valid["NATENVIR_FLAG"] + valid["HELPBLK_FLAG"] + valid["HELPSICK_FLAG"] + valid["HELPPOOR_FLAG"]

# test
test['HELPPOOR_FLAG'] = test['HELPPOOR']
test["HELPPOOR_FLAG"] = test["HELPPOOR_FLAG"].replace(hlppr_ec)
test["HELPPOOR_FLAG"] = test["HELPPOOR_FLAG"].astype(int)

test["HELPSICK_FLAG"] = test["HELPSICK"]
test["HELPSICK_FLAG"] = test["HELPSICK_FLAG"].replace(hlpsck_ec)
test["HELPSICK_FLAG"] = test["HELPSICK_FLAG"].astype(int)

test["HELPBLK_FLAG"] = test["HELPBLK"]
test["HELPBLK_FLAG"] = test["HELPBLK_FLAG"].replace(hlpblk_ec)
test["HELPBLK_FLAG"] = test["HELPBLK_FLAG"].astype(int)

test["ABANY_FLAG"] = test["ABANY"]
test["ABANY_FLAG"] = test["ABANY_FLAG"].replace(abany_ec)
test["ABANY_FLAG"] = test["ABANY_FLAG"].astype(int)

test["NATENVIR_FLAG"] = test["NATENVIR"]
test["NATENVIR_FLAG"] = test["NATENVIR_FLAG"].replace(natenv_ec)
test["NATENVIR_FLAG"] = test["NATENVIR_FLAG"].astype(int)

test["LIB_SCORE"] = test["ABANY_FLAG"] + test["NATENVIR_FLAG"] + test["HELPBLK_FLAG"] + test["HELPSICK_FLAG"] + test["HELPPOOR_FLAG"]

## CONF IN INSTITUTIONS

#train
train["CONFED_FLAG"] = train["CONFED"]
conf_ec = {
    'Only Some':'2',
    'Hardly Any': '1',
    'A Great Deal': '3'
}
train["CONFED_FLAG"] = train["CONFED_FLAG"].replace(conf_ec)
train["CONFED_FLAG"] = train["CONFED_FLAG"].astype(int)


train["CONARMY_FLAG"] = train["CONARMY"]
train["CONARMY_FLAG"] = train["CONARMY_FLAG"].replace(conf_ec)
train["CONARMY_FLAG"] = train["CONARMY_FLAG"].astype(int)


train["CONJUDGE_FLAG"] = train["CONJUDGE"]
train["CONJUDGE_FLAG"] = train["CONJUDGE_FLAG"].replace(conf_ec)
train["CONJUDGE_FLAG"] = train["CONJUDGE_FLAG"].astype(int)


train["CONF_SCORE"] = train["CONFED_FLAG"] + train["CONARMY_FLAG"] + train["CONJUDGE_FLAG"]

#valid
valid["CONFED_FLAG"] = valid["CONFED"]
valid["CONFED_FLAG"] = valid["CONFED_FLAG"].replace(conf_ec)
valid["CONFED_FLAG"] = valid["CONFED_FLAG"].astype(int)


valid["CONARMY_FLAG"] = valid["CONARMY"]
valid["CONARMY_FLAG"] = valid["CONARMY_FLAG"].replace(conf_ec)
valid["CONARMY_FLAG"] = valid["CONARMY_FLAG"].astype(int)


valid["CONJUDGE_FLAG"] = valid["CONJUDGE"]
valid["CONJUDGE_FLAG"] = valid["CONJUDGE_FLAG"].replace(conf_ec)
valid["CONJUDGE_FLAG"] = valid["CONJUDGE_FLAG"].astype(int)


valid["CONF_SCORE"] = valid["CONFED_FLAG"] + valid["CONARMY_FLAG"] + valid["CONJUDGE_FLAG"]

#test
test["CONFED_FLAG"] = test["CONFED"]
test["CONFED_FLAG"] = test["CONFED_FLAG"].replace(conf_ec)
test["CONFED_FLAG"] = test["CONFED_FLAG"].astype(int)


test["CONARMY_FLAG"] = test["CONARMY"]
test["CONARMY_FLAG"] = test["CONARMY_FLAG"].replace(conf_ec)
test["CONARMY_FLAG"] = test["CONARMY_FLAG"].astype(int)


test["CONJUDGE_FLAG"] = test["CONJUDGE"]
test["CONJUDGE_FLAG"] = test["CONJUDGE_FLAG"].replace(conf_ec)
test["CONJUDGE_FLAG"] = test["CONJUDGE_FLAG"].astype(int)


test["CONF_SCORE"] = test["CONFED_FLAG"] + test["CONARMY_FLAG"] + test["CONJUDGE_FLAG"]

train.drop(
    columns=['CASEID', 'ABANY', 'HELPPOOR', 'HELPSICK', 'HELPBLK', 'CONFED', 'CONARMY', 'CONJUDGE',
             'NATENVIR',  'NATENVIR_FLAG', 'HELPPOOR_FLAG', 'HELPSICK_FLAG', 'HELPBLK_FLAG', 'ABANY_FLAG',
             'CONFED_FLAG', 'CONARMY_FLAG', 'CONJUDGE_FLAG'], axis=1, inplace=True)

valid.drop(
    columns=['CASEID', 'ABANY', 'HELPPOOR', 'HELPSICK', 'HELPBLK', 'CONFED', 'CONARMY', 'CONJUDGE',
             'NATENVIR',  'NATENVIR_FLAG', 'HELPPOOR_FLAG', 'HELPSICK_FLAG', 'HELPBLK_FLAG', 'ABANY_FLAG',
             'CONFED_FLAG', 'CONARMY_FLAG', 'CONJUDGE_FLAG'], axis=1, inplace=True)

test.drop(
    columns=['CASEID', 'ABANY', 'HELPPOOR', 'HELPSICK', 'HELPBLK', 'CONFED', 'CONARMY', 'CONJUDGE',
             'NATENVIR',  'NATENVIR_FLAG', 'HELPPOOR_FLAG', 'HELPSICK_FLAG', 'HELPBLK_FLAG', 'ABANY_FLAG',
             'CONFED_FLAG', 'CONARMY_FLAG', 'CONJUDGE_FLAG'], axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(train, cat_th=5)
train.columns

for col in num_cols:
    target_summary_with_num(train, "IS_LEFT", col)
for col in cat_cols:
    target_summary_with_cat(train, "IS_LEFT", col)

train_yes_left = train.loc[train['IS_LEFT'] == 1, :]
smple = train_yes_left.sample(n=(len(train.loc[train['IS_LEFT'] == 0]) - len(train.loc[train['IS_LEFT'] == 1])),
                              replace=True, random_state=42)
train_new = pd.concat([train, smple], ignore_index=True)

train_new.IS_LEFT.value_counts()

train_new.columns
train_new.info()

for col in train_new.columns:
    if train_new[col].dtypes == 'object':
        print(str(col))
        print(train_new[col].nunique())

binary_cols = [col for col in train_new.columns if col != 'IS_LEFT' and train_new[col].nunique() == 2]
train_new = one_hot_encoder(train_new, binary_cols, drop_first=True)
valid = one_hot_encoder(valid, binary_cols, drop_first=True)
test = one_hot_encoder(test, binary_cols, drop_first=True)

# train_new['DEGREE'].unique()
# degree_oe = {
#     'Lt High School': '0',
#     'High School': '1',
#     'Junior College': '2',
#     'Bachelor': '3',
#     'Graduate': '4'
# }
# train_new['DEGREE'] = train_new['DEGREE'].replace(degree_oe)
# train_new['DEGREE'] = train_new['DEGREE'].astype(int)
#
# valid['DEGREE'] = valid['DEGREE'].replace(degree_oe)
# valid['DEGREE'] = valid['DEGREE'].astype(int)
#
# test['DEGREE'] = test['DEGREE'].replace(degree_oe)
# test['DEGREE'] = test['DEGREE'].astype(int)
#
# train_new['MARITAL'].unique()
#
# train_new['CLASS'].unique()
# class_oe = {
#     'Lower Class': '0',
#     'Working Class': '1',
#     'Middle Class': '2',
#     'Upper Class': '3'
# }
# train_new['CLASS'] = train_new['CLASS'].replace(class_oe)
# train_new['CLASS'] = train_new['CLASS'].astype(int)
#
# valid['CLASS'] = valid['CLASS'].replace(class_oe)
# valid['CLASS'] = valid['CLASS'].astype(int)
#
# test['CLASS'] = test['CLASS'].replace(class_oe)
# test['CLASS'] = test['CLASS'].astype(int)

cat_cols = ['DEGREE', 'MARITAL', 'CLASS']
train_new = one_hot_encoder(train_new, cat_cols, drop_first=False)
valid = one_hot_encoder(valid, cat_cols, drop_first=False)
test = one_hot_encoder(test, cat_cols, drop_first=False)

train_new.shape[1] == valid.shape[1]
train_new.shape[1] == test.shape[1]

## MinMaxScaling
mms = MinMaxScaler(feature_range=(0,1))

mms.fit(train_new[['LIB_SCORE']])
scaled = mms.transform(train_new[['LIB_SCORE']])
train_new["LIB_SCORE"] = scaled

scaled = mms.transform(valid[['LIB_SCORE']])
valid['LIB_SCORE'] = scaled

scaled = mms.transform(test[['LIB_SCORE']])
test['LIB_SCORE'] = scaled

mms.fit(train_new[['CONF_SCORE']])
scaled = mms.transform(train_new[['CONF_SCORE']])
train_new["CONF_SCORE"] = scaled

scaled = mms.transform(valid[['CONF_SCORE']])
valid['CONF_SCORE'] = scaled

scaled = mms.transform(test[['CONF_SCORE']])
test['CONF_SCORE'] = scaled

mms.fit(train_new[['AGE']])
scaled = mms.transform(train_new[['AGE']])
train_new["AGE"] = scaled

scaled = mms.transform(valid[['AGE']])
valid["AGE"] = scaled

scaled = mms.transform(test[['AGE']])
test["AGE"] = scaled

mms.fit(train_new[['CHILDS']])
scaled = mms.transform(train_new[['CHILDS']])
train_new["CHILDS"] = scaled

scaled = mms.transform(valid[['CHILDS']])
valid["CHILDS"] = scaled

scaled = mms.transform(test[['CHILDS']])
test["CHILDS"] = scaled

y_test = test["IS_LEFT"]
X_test = test.drop(["YEAR","IS_LEFT", "POLVIEWS"], axis=1)

y_val = valid["IS_LEFT"]
X_val = valid.drop(["YEAR","IS_LEFT", "POLVIEWS"], axis=1)

y_train = train_new["IS_LEFT"]
X_train = train_new.drop(["YEAR","IS_LEFT", "POLVIEWS"], axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV, validation_curve, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# LogisticRegression
lr = LogisticRegression(random_state=42).fit(X_train, y_train)

y_pred = lr.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .673
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .632
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .7
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .664

from sklearn.model_selection import PredefinedSplit

X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

len(X) == (X_train.shape[0] + X_val.shape[0])
len(y) == (y_train.shape[0] + y_val.shape[0])

split_index = [0 if x in X_val.index else -1 for x in X.index]
pds = PredefinedSplit(test_fold = split_index)
pds.get_n_splits()

lr = LogisticRegression(random_state=42)

weights = np.linspace(0.0,0.99,200)
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

gridsearch = GridSearchCV(estimator= lr,
                          param_grid= param_grid,
                          cv=pds,
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X, y)

plt.figure(figsize=(12,8))
weight_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(data=weight_data, x='weight', y='score', ci=False)
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
plt.show()

weight_data.sort_values(by='score', ascending=False).head(1)

lr = LogisticRegression(random_state=42, class_weight={0: 1-.637, 1: .637})
lr.fit(X_train, y_train)
# for mental check
y_pred = lr.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .614
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .55
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .914
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .686

lr.fit(X, y)
# test score
y_pred = lr.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .63
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .562
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}") # .852
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .678


# DecisionTreeClassifier
cft = DecisionTreeClassifier(random_state=42)
cft.fit(X_train, y_train)

y_pred = cft.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .578
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .545
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .519
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") #.532

cft.get_params()

cft_params = {
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_split': [20, 23, 25, 30, 35],
    'min_samples_leaf': [10, 15, 20, 25]
}

grid_search = GridSearchCV(
    cft,
    cft_params,
    cv=pds,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X, y)

grid_search.best_params_

cft_fin = DecisionTreeClassifier(**grid_search.best_params_, random_state=42).fit(X, y)

y_pred = cft_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") #.66
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") #.632
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}") #.609
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") #.621

# RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .608
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .577
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .568
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .573


rf_params = {
    "max_depth": [3, 5, 8, 12, None],
    "max_features": [7, 10, 12, 15, 17],
    "min_samples_split": [8, 15, 20, 25],
    "n_estimators": [300, 500, 800, 1000]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    rf_params,
    cv=pds,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

grid_search.best_params_

rf_fin = RandomForestClassifier(**grid_search.best_params_, random_state=42).fit(X, y)

y_pred = rf_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .655
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .616
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}") # .646
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .631


lgbm = LGBMClassifier(random_state=42)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .646
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .614
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .634
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .623

lgbm.get_params()

lgbm_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [300, 500, 800, 1000],
    'colsample_bytree': [0.5, 0.7, 0.9, 1],
    'max_depth' : [-1, 3, 5, 8, 12],
    'num_leaves': [num for num in range(16, 56, 8)],
    'min_child_samples': [num for num in range(5, 45, 5)]
}

lgbm = LGBMClassifier(random_state=42)

grid_search = GridSearchCV(
    lgbm,
    lgbm_params,
    cv=pds,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

grid_search.best_params_

lgbm_fin = LGBMClassifier(**grid_search.best_params_, random_state=42).fit(X, y)

y_pred = lgbm_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .66
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .621
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")  # .654
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .637

from sklearn.inspection import permutation_importance

r_multi = permutation_importance(lgbm_fin, X_train, y_train,
                                 n_repeats=30,
                                 random_state=0,
                                 scoring=['accuracy', 'precision', 'recall', 'f1'])

for metric in r_multi:
     print(f"{metric}")
     r = r_multi[metric]
     for i in r.importances_mean.argsort()[::-1]:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
             print(f"    {X_train.columns[i]:<8}"
                   f" {r.importances_mean[i]:.3f}"
                   f" +/- {r.importances_std[i]:.3f}")

r_multi = permutation_importance(lgbm_fin, X_test, y_test,
                                 n_repeats=30,
                                 random_state=0,
                                 scoring=['accuracy', 'precision', 'recall', 'f1'])

for metric in r_multi:
     print(f"{metric}")
     r = r_multi[metric]
     for i in r.importances_mean.argsort()[::-1]:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
             print(f"    {X_train.columns[i]:<8}"
                   f" {r.importances_mean[i]:.3f}"
                   f" +/- {r.importances_std[i]:.3f}")

from pdpbox import pdp

feature_names = X_test.columns.tolist()

pdp_age = pdp.pdp_isolate(model=lgbm_fin, dataset=X_test, \
                          model_features=feature_names, feature='AGE')

pdp.pdp_plot(pdp_age, 'Age')
plt.show()

import shap

explainer = shap.TreeExplainer(lgbm_fin)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)

explainer = shap.TreeExplainer(lgbm_fin)
shap_values = explainer.shap_values(X_test)
shap.dependence_plot('LIB_SCORE', shap_values[1], X_test)


