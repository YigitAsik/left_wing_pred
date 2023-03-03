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

df.loc[df["POLVIEWS"] == "Extremely Liberal", "POLVIEWS"] = "1"
df.loc[df["POLVIEWS"] == "Liberal", "POLVIEWS"] = "2"
df.loc[df["POLVIEWS"] == "Slightly Liberal", "POLVIEWS"] = "3"
df.loc[df["POLVIEWS"] == "Moderate", "POLVIEWS"] = "4"
df.loc[df["POLVIEWS"] == "Slightly Conservative", "POLVIEWS"] = "5"
df.loc[df["POLVIEWS"] == "Conservative", "POLVIEWS"] = "6"
df.loc[df["POLVIEWS"] == "Extrmly Conservative", "POLVIEWS"] = "7"
df["POLVIEWS"] = df["POLVIEWS"].astype(int)

df.drop(df.loc[df["POLVIEWS"] == 4, :].index, inplace=True)

df.POLVIEWS.unique()

train, rem = train_test_split(df, train_size=.7, stratify=df.POLVIEWS, random_state=42)

valid, test = train_test_split(rem, test_size=.5, stratify=rem.POLVIEWS, random_state=42)


train.head()
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

for col in num_cols:
    fig = plt.figure(figsize=(9,8))
    g = sns.barplot(data=train,
                    x="POLVIEWS", y="AGE",
                    palette="viridis", ci=False)
    show_values(g)
    fig.show()

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

# for col in cat_cols:
#     print('Col name:' + str(col))
#     cat_summary(train, col)

## LIB_SCORE

# train
train['HELPPOOR_FLAG'] = train['HELPPOOR']
train.loc[train["HELPPOOR_FLAG"] == "Agree With Both", "HELPPOOR_FLAG"] = "2"
train.loc[train["HELPPOOR_FLAG"] == "Govt Action", "HELPPOOR_FLAG"] = "3"
train.loc[train["HELPPOOR_FLAG"] == "People Help Selves", "HELPPOOR_FLAG"] = "1"
train["HELPPOOR_FLAG"] = train["HELPPOOR_FLAG"].astype(int)

train["HELPSICK_FLAG"] = train["HELPSICK"]
train.loc[train["HELPSICK_FLAG"] == "Agree With Both", "HELPSICK_FLAG"] = "2"
train.loc[train["HELPSICK_FLAG"] == "Govt Should Help", "HELPSICK_FLAG"] = "3"
train.loc[train["HELPSICK_FLAG"] == "People Help Selves", "HELPSICK_FLAG"] = "1"
train["HELPSICK_FLAG"] = train["HELPSICK_FLAG"].astype(int)

train["HELPBLK_FLAG"] = train["HELPBLK"]
train.loc[train["HELPBLK_FLAG"] == "Agree With Both", "HELPBLK_FLAG"] = "2"
train.loc[train["HELPBLK_FLAG"] == "Govt Help Blks", "HELPBLK_FLAG"] = "3"
train.loc[train["HELPBLK_FLAG"] == "No Special Treatment", "HELPBLK_FLAG"] = "1"
train["HELPBLK_FLAG"] = train["HELPBLK_FLAG"].astype(int)

train["ABANY_FLAG"] = train["ABANY"]
train.loc[train["ABANY_FLAG"] == "DK", "ABANY_FLAG"] = "2"
train.loc[train["ABANY_FLAG"] == "Yes", "ABANY_FLAG"] = "3"
train.loc[train["ABANY_FLAG"] == "No", "ABANY_FLAG"] = "1"
train["ABANY_FLAG"] = train["ABANY_FLAG"].astype(int)

train["NATENVIR_FLAG"] = train["NATENVIR"]
train.loc[train["NATENVIR_FLAG"] == "About Right", "NATENVIR_FLAG"] = "2"
train.loc[train["NATENVIR_FLAG"] == "Too Much", "NATENVIR_FLAG"] = "3"
train.loc[train["NATENVIR_FLAG"] == "Too Little", "NATENVIR_FLAG"] = "1"
train["NATENVIR_FLAG"] = train["NATENVIR_FLAG"].astype(int)

train["LIB_SCORE"] = train["ABANY_FLAG"] + train["NATENVIR_FLAG"] + train["HELPBLK_FLAG"] + train["HELPSICK_FLAG"] + train["HELPPOOR_FLAG"]

# valid
valid['HELPPOOR_FLAG'] = valid['HELPPOOR']
valid.loc[valid["HELPPOOR_FLAG"] == "Agree With Both", "HELPPOOR_FLAG"] = "2"
valid.loc[valid["HELPPOOR_FLAG"] == "Govt Action", "HELPPOOR_FLAG"] = "3"
valid.loc[valid["HELPPOOR_FLAG"] == "People Help Selves", "HELPPOOR_FLAG"] = "1"
valid["HELPPOOR_FLAG"] = valid["HELPPOOR_FLAG"].astype(int)

valid["HELPSICK_FLAG"] = valid["HELPSICK"]
valid.loc[valid["HELPSICK_FLAG"] == "Agree With Both", "HELPSICK_FLAG"] = "2"
valid.loc[valid["HELPSICK_FLAG"] == "Govt Should Help", "HELPSICK_FLAG"] = "3"
valid.loc[valid["HELPSICK_FLAG"] == "People Help Selves", "HELPSICK_FLAG"] = "1"
valid["HELPSICK_FLAG"] = valid["HELPSICK_FLAG"].astype(int)

valid["HELPBLK_FLAG"] = valid["HELPBLK"]
valid.loc[valid["HELPBLK_FLAG"] == "Agree With Both", "HELPBLK_FLAG"] = "2"
valid.loc[valid["HELPBLK_FLAG"] == "Govt Help Blks", "HELPBLK_FLAG"] = "3"
valid.loc[valid["HELPBLK_FLAG"] == "No Special Treatment", "HELPBLK_FLAG"] = "1"
valid["HELPBLK_FLAG"] = valid["HELPBLK_FLAG"].astype(int)

valid["ABANY_FLAG"] = valid["ABANY"]
valid.loc[valid["ABANY_FLAG"] == "DK", "ABANY_FLAG"] = "2"
valid.loc[valid["ABANY_FLAG"] == "Yes", "ABANY_FLAG"] = "3"
valid.loc[valid["ABANY_FLAG"] == "No", "ABANY_FLAG"] = "1"
valid["ABANY_FLAG"] = valid["ABANY_FLAG"].astype(int)

valid["NATENVIR_FLAG"] = valid["NATENVIR"]
valid.loc[valid["NATENVIR_FLAG"] == "About Right", "NATENVIR_FLAG"] = "2"
valid.loc[valid["NATENVIR_FLAG"] == "Too Much", "NATENVIR_FLAG"] = "3"
valid.loc[valid["NATENVIR_FLAG"] == "Too Little", "NATENVIR_FLAG"] = "1"
valid["NATENVIR_FLAG"] = valid["NATENVIR_FLAG"].astype(int)

valid["LIB_SCORE"] = valid["ABANY_FLAG"] + valid["NATENVIR_FLAG"] + valid["HELPBLK_FLAG"] + valid["HELPSICK_FLAG"] + valid["HELPPOOR_FLAG"]

# test
test['HELPPOOR_FLAG'] = test['HELPPOOR']
test.loc[test["HELPPOOR_FLAG"] == "Agree With Both", "HELPPOOR_FLAG"] = "2"
test.loc[test["HELPPOOR_FLAG"] == "Govt Action", "HELPPOOR_FLAG"] = "3"
test.loc[test["HELPPOOR_FLAG"] == "People Help Selves", "HELPPOOR_FLAG"] = "1"
test["HELPPOOR_FLAG"] = test["HELPPOOR_FLAG"].astype(int)

test["HELPSICK_FLAG"] = test["HELPSICK"]
test.loc[test["HELPSICK_FLAG"] == "Agree With Both", "HELPSICK_FLAG"] = "2"
test.loc[test["HELPSICK_FLAG"] == "Govt Should Help", "HELPSICK_FLAG"] = "3"
test.loc[test["HELPSICK_FLAG"] == "People Help Selves", "HELPSICK_FLAG"] = "1"
test["HELPSICK_FLAG"] = test["HELPSICK_FLAG"].astype(int)

test["HELPBLK_FLAG"] = test["HELPBLK"]
test.loc[test["HELPBLK_FLAG"] == "Agree With Both", "HELPBLK_FLAG"] = "2"
test.loc[test["HELPBLK_FLAG"] == "Govt Help Blks", "HELPBLK_FLAG"] = "3"
test.loc[test["HELPBLK_FLAG"] == "No Special Treatment", "HELPBLK_FLAG"] = "1"
test["HELPBLK_FLAG"] = test["HELPBLK_FLAG"].astype(int)

test["ABANY_FLAG"] = test["ABANY"]
test.loc[test["ABANY_FLAG"] == "DK", "ABANY_FLAG"] = "2"
test.loc[test["ABANY_FLAG"] == "Yes", "ABANY_FLAG"] = "3"
test.loc[test["ABANY_FLAG"] == "No", "ABANY_FLAG"] = "1"
test["ABANY_FLAG"] = test["ABANY_FLAG"].astype(int)

test["NATENVIR_FLAG"] = test["NATENVIR"]
test.loc[test["NATENVIR_FLAG"] == "About Right", "NATENVIR_FLAG"] = "2"
test.loc[test["NATENVIR_FLAG"] == "Too Much", "NATENVIR_FLAG"] = "3"
test.loc[test["NATENVIR_FLAG"] == "Too Little", "NATENVIR_FLAG"] = "1"
test["NATENVIR_FLAG"] = test["NATENVIR_FLAG"].astype(int)

test["LIB_SCORE"] = test["ABANY_FLAG"] + test["NATENVIR_FLAG"] + test["HELPBLK_FLAG"] + test["HELPSICK_FLAG"] + test["HELPPOOR_FLAG"]

## CONF IN INSTITUTIONS

#train
train["CONFED_FLAG"] = train["CONFED"]
train.loc[train["CONFED_FLAG"] == "Only Some", "CONFED_FLAG"] = "2"
train.loc[train["CONFED_FLAG"] == "Hardly Any", "CONFED_FLAG"] = "1"
train.loc[train["CONFED_FLAG"] == "A Great Deal", "CONFED_FLAG"] = "3"
train["CONFED_FLAG"] = train["CONFED_FLAG"].astype(int)

train["CONARMY_FLAG"] = train["CONARMY"]
train.loc[train["CONARMY_FLAG"] == "Only Some", "CONARMY_FLAG"] = "2"
train.loc[train["CONARMY_FLAG"] == "Hardly Any", "CONARMY_FLAG"] = "1"
train.loc[train["CONARMY_FLAG"] == "A Great Deal", "CONARMY_FLAG"] = "3"
train["CONARMY_FLAG"] = train["CONARMY_FLAG"].astype(int)

train["CONJUDGE_FLAG"] = train["CONJUDGE"]
train.loc[train["CONJUDGE_FLAG"] == "Only Some", "CONJUDGE_FLAG"] = "2"
train.loc[train["CONJUDGE_FLAG"] == "Hardly Any", "CONJUDGE_FLAG"] = "1"
train.loc[train["CONJUDGE_FLAG"] == "A Great Deal", "CONJUDGE_FLAG"] = "3"
train["CONJUDGE_FLAG"] = train["CONJUDGE_FLAG"].astype(int)

train["CONF_SCORE"] = train["CONFED_FLAG"] + train["CONARMY_FLAG"] + train["CONJUDGE_FLAG"]

#valid
valid["CONFED_FLAG"] = valid["CONFED"]
valid.loc[valid["CONFED_FLAG"] == "Only Some", "CONFED_FLAG"] = "2"
valid.loc[valid["CONFED_FLAG"] == "Hardly Any", "CONFED_FLAG"] = "1"
valid.loc[valid["CONFED_FLAG"] == "A Great Deal", "CONFED_FLAG"] = "3"
valid["CONFED_FLAG"] = valid["CONFED_FLAG"].astype(int)

valid["CONARMY_FLAG"] = valid["CONARMY"]
valid.loc[valid["CONARMY_FLAG"] == "Only Some", "CONARMY_FLAG"] = "2"
valid.loc[valid["CONARMY_FLAG"] == "Hardly Any", "CONARMY_FLAG"] = "1"
valid.loc[valid["CONARMY_FLAG"] == "A Great Deal", "CONARMY_FLAG"] = "3"
valid["CONARMY_FLAG"] = valid["CONARMY_FLAG"].astype(int)

valid["CONJUDGE_FLAG"] = valid["CONJUDGE"]
valid.loc[valid["CONJUDGE_FLAG"] == "Only Some", "CONJUDGE_FLAG"] = "2"
valid.loc[valid["CONJUDGE_FLAG"] == "Hardly Any", "CONJUDGE_FLAG"] = "1"
valid.loc[valid["CONJUDGE_FLAG"] == "A Great Deal", "CONJUDGE_FLAG"] = "3"
valid["CONJUDGE_FLAG"] = valid["CONJUDGE_FLAG"].astype(int)

valid["CONF_SCORE"] = valid["CONFED_FLAG"] + valid["CONARMY_FLAG"] + valid["CONJUDGE_FLAG"]

#test
test["CONFED_FLAG"] = test["CONFED"]
test.loc[test["CONFED_FLAG"] == "Only Some", "CONFED_FLAG"] = "2"
test.loc[test["CONFED_FLAG"] == "Hardly Any", "CONFED_FLAG"] = "1"
test.loc[test["CONFED_FLAG"] == "A Great Deal", "CONFED_FLAG"] = "3"
test["CONFED_FLAG"] = test["CONFED_FLAG"].astype(int)

test["CONARMY_FLAG"] = test["CONARMY"]
test.loc[test["CONARMY_FLAG"] == "Only Some", "CONARMY_FLAG"] = "2"
test.loc[test["CONARMY_FLAG"] == "Hardly Any", "CONARMY_FLAG"] = "1"
test.loc[test["CONARMY_FLAG"] == "A Great Deal", "CONARMY_FLAG"] = "3"
test["CONARMY_FLAG"] = test["CONARMY_FLAG"].astype(int)

test["CONJUDGE_FLAG"] = test["CONJUDGE"]
test.loc[test["CONJUDGE_FLAG"] == "Only Some", "CONJUDGE_FLAG"] = "2"
test.loc[test["CONJUDGE_FLAG"] == "Hardly Any", "CONJUDGE_FLAG"] = "1"
test.loc[test["CONJUDGE_FLAG"] == "A Great Deal", "CONJUDGE_FLAG"] = "3"
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
train_new = train.append(smple, ignore_index=True)

train_new.IS_LEFT.value_counts()

train_new.columns
cat_cols = ['SEX', 'RACE', 'DEGREE', 'MARITAL', 'CLASS']
train_new = one_hot_encoder(train_new, cat_cols, drop_first=True)
valid = one_hot_encoder(valid, cat_cols, drop_first=True)
test = one_hot_encoder(test, cat_cols, drop_first=True)


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

Val = valid.copy()
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
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .671
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .631
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .695
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .661

# weight = np.linspace(0.0, 0.99, 200)

from sklearn.model_selection import PredefinedSplit

dff = train_new.copy()
dff = dff.append(Val, ignore_index=True)

dff.iloc[-1, :]
Val.iloc[-1, :]

split_index = [-1]*len(train_new) + [0]*len(Val)
pds = PredefinedSplit(test_fold = split_index)

X = dff.drop(["YEAR", "IS_LEFT", "POLVIEWS"], axis=1)
y = dff['IS_LEFT']

# DecisionTreeClassifier
cft = DecisionTreeClassifier(random_state=42)
cft.fit(X_train, y_train)

y_pred = cft.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .555
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .519
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .502
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") #.51

cft.get_params()

cft_params = {
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_split': [20, 23, 25, 30, 35],
    'min_samples_leaf': [10, 15]
}

grid_search = GridSearchCV(
    cft,
    cft_params,
    cv=pds,
    n_jobs=-1
)

grid_search.fit(X, y)

grid_search.best_params_

cft_fin = DecisionTreeClassifier(**grid_search.best_params_, random_state=42).fit(X_train, y_train)

y_pred = cft_fin.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .671
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .606
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") #.823
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") #.698
# RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .608
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .575
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .584
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .58


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
    n_jobs=-1
)

grid_search.fit(X, y)

grid_search.best_params_

rf_fin = RandomForestClassifier(**grid_search.best_params_, random_state=42).fit(X_train, y_train)

y_pred = rf_fin.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .665
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .621
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .708
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .662

lgbm = LGBMClassifier(random_state=42)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .639
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .605
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .630
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .617

lgbm.get_params()

lgbm_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [300, 500, 800, 1000],
    'colsample_bytree': [0.5, 0.7, 1],
    'max_depth' : [-1, 3, 5, 8, 12],
    'num_leaves': [20, 30, 40, 50, 70],
    'min_child_samples': [15, 25, 30, 40, 50, 70]
}

lgbm = LGBMClassifier(random_state=42)

grid_search = GridSearchCV(
    lgbm,
    lgbm_params,
    cv=pds,
    n_jobs=-1
)

grid_search.fit(X, y)

grid_search.best_params_

lgbm_fin = LGBMClassifier(**grid_search.best_params_, random_state=42).fit(X_train, y_train)

y_pred = lgbm_fin.predict(X_val)
print(f"Accuracy: {round(accuracy_score(y_pred, y_val), 3)}") # .639
print(f"Recall: {round(recall_score(y_pred,y_val),3)}") # .605
print(f"Precision: {round(precision_score(y_pred,y_val), 3)}") # .63
print(f"F1: {round(f1_score(y_pred,y_val), 3)}") # .617


# Test scores
y_pred = cft_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .664
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .605
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")  # .761
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .674

tree_rules = export_text(cft_fin, feature_names=list(X_train.columns))
print(tree_rules)

fig = plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_test, y_pred, labels=cft_fin.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cft_fin.classes_)
disp.plot()
plt.show()

import pydotplus
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cft_fin, col_names=X_train.columns, file_name='cft_lw.png')

cft_fin.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cft_fin, X_train, 5)


y_pred = rf_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .645
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .604
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}") # .646
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .624

y_pred = lgbm_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}") # .653
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # .619
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")  # .621
print(f"F1: {round(f1_score(y_pred,y_test), 3)}") # .62