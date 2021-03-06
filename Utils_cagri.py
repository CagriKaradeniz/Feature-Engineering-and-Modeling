import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import train_test_split
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import Utils_cagri as util
import random
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

models = [LogisticRegression,KNeighborsClassifier,SVC,MLPClassifier,
            DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier,
            XGBClassifier,LGBMClassifier,CatBoostClassifier]

def dataset_yukle(dataset):
    return pd.read_csv(dataset+".csv")

def degisken_tiplerine_ayirma(data,cat_th,car_th):
   """
   Veri:data parametresi ili fonksiyona girilen verinin de??i??kenlerin s??n??fland??r??lmas??.
   Parameters
   ----------
   data: pandas.DataFrame
   ????lem yap??lacak veri seti

   cat_th:int
   categoric de??i??ken threshold de??eri

   car_th:int
   Cardinal de??i??kenler i??in threshold de??eri

   Returns
   -------
    cat_deg:list
    categorik de??i??ken listesi
    num_deg:list
    numeric de??i??ken listesi
    car_deg:list
    categoric ama cardinal de??i??ken listesi

   Examples
   -------
    df = dataset_yukle("breast_cancer")
    cat,num,car=degisken_tiplerine_ayirma(df,10,20)
   Notes
   -------
    cat_deg + num_deg + car_deg = toplam de??i??ken say??s??

   """


   num_but_cat=[i for i in data.columns if data[i].dtypes !="O" and data[i].nunique() < cat_th]

   car_deg=[i for i in data.columns if data[i].dtypes == "O" and data[i].nunique() > car_th]

   num_deg=[i for i in data.columns if data[i].dtypes !="O" and i not in num_but_cat]

   cat_deg = [i for i in data.columns if data[i].dtypes == "O" and i not in car_deg]

   cat_deg = cat_deg+num_but_cat

   print(f"Dataset kolon/de??i??ken say??s??: {data.shape[1]}")
   print(f"Dataset sat??r/veri say??s??: {data.shape[0]}")
   print("********************************************")
   print(f"Datasetin numeric de??i??ken say??s??: {len(num_deg)}")
   print(f"Datasetin numeric de??i??kenler: {num_deg}")
   print("********************************************")
   print(f"Datasetin categoric de??i??ken say??s??: {len(cat_deg)}")
   print(f"Datasetin categoric de??i??kenler: {cat_deg}")
   print("********************************************")
   print(f"Datasetin cardinal de??i??ken say??s??: {len(car_deg)}")
   print(f"Datasetin cardinal de??i??kenler: {car_deg}")
   print("********************************************")

   return cat_deg,num_deg,car_deg

def categoric_ozet(data,degisken,plot=False,null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan categoric de??i??kenlerin de??i??ken tiplerinin say??s??n?? ve totale kar???? oran??n?? bulur.
    Ayr??ca iste??e ba??l?? olarak de??i??ken da????l??m??n??n grafi??ini ve de??i??ken i??inde bulunan null say??s??n?? ????kart??r.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric de??i??kenin bulundu??u dataset.
    degisken:String
    Categoric de??i??ken ismi.
    plot:bool
    Fonksiyonda categoric de??i??ken da????l??m??n??n grafi??ini ??izdirmek i??in opsiyonel ??zellik.
    null_control:bool
    Fonksiyonda de??i??ken i??inde null de??er kontol?? i??in opsiyonel ??zellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique de??i??kenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """

    print(pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)}))
    tablo=pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)})
    print("##########################################")
    if plot:
        sns.countplot(x=data[degisken], data=data)
        plt.show()
    if null_control:
        print(f"Null veri say??s??: {data[degisken].isnull().sum()}")

    return tablo
def dataset_ozet(data, head=5):
    print("##################### Shape #####################")
    print(f"Sat??r say??s??: {data.shape[0]}")
    print(f"Kolon say??s??: {data.shape[1]}")

    print("##################### Types #####################")
    print(data.dtypes)

    print("##################### Head #####################")
    print(data.head(head))

    print("##################### Tail #####################")
    print(data.tail(head))

    print("##################### NA Kontrol?? #####################")
    print(data.isnull().sum())

    print("##################### Quantiles #####################")
    print(data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("##################### Describe Tablosu #####################")
    print(data.describe().T)

def outlier_threshold(data,degisken):
    Q1=data[degisken].quantile(0.01)
    Q3=data[degisken].quantile(0.99)
    Q_Inter_Range=Q3-Q1
    alt_limit=Q1-1.5*Q_Inter_Range
    ust_limit=Q3+1.5*Q_Inter_Range
    return alt_limit,ust_limit
def threshold_degisimi(data,degisken):
    alt_limit,ust_limit=outlier_threshold(data,degisken)
    data[data[degisken]<alt_limit]=alt_limit
    data[data[degisken]>ust_limit]=ust_limit
    return data

def data_hazirlama(data):
    data.dropna(inplace=True)
    data = data[~data["Invoice"].str.contains("C", na=False)]
    data = data[data["Quantity"] > 0]
    data = data[data["Price"] > 0]
    threshold_degisimi(data, "Quantity")
    threshold_degisimi(data, "Price")
    return data

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    #print(product_name)
    return product_name
def kural_olustur_kitap(data):
    frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def kural_olustur(data, id=True,country="Germany"):
    data = data[data['Country'] == country]
    data = create_invoice_product_df(data, id)
    frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

def kullan??c??Based_dataolustur(data1,data2):

    df = data1.merge(data2, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

def item_based_recommender(film, data):
    movie_name = data[film]
    return data.corrwith(movie_name).sort_values(ascending=False).head(10)

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

def A_B_Testing(data1,data2,degisken):
    """

    Parameters
    ----------
    data1
    data2
    degisken

    Returns
    -------

    """
    p_ths=0.05
    t_stat1,p_value1=shapiro(data1[degisken])
    t_stat2, p_value2 = shapiro(data2[degisken])
    t_stat3, p_value3 = levene(data1[degisken],data2[degisken])
    if p_value1 > p_ths and p_value2 > p_ths and p_value3 > p_ths:
        t_stat, p_value = ttest_ind(data1[degisken], data2[degisken], equal_var=True)
        if p_value > p_ths:
            print(f"{p_value} bu say?? {p_ths}'den b??y??kt??r, H0 reddedilemez. ??ki data da ayn?? da????l??mdad??r")
        else:
            print(f"{p_value} bu say?? {p_ths}'den k??????kt??r, H0 red edilir. ??ki Data aras??nda da????l??m fark?? vard??r")
    elif p_value1 > p_ths and p_value2 > p_ths and p_value3 < p_ths:
        t_stat, p_value = ttest_ind(data1[degisken], data2[degisken], equal_var=True)
        if p_value > p_ths:
            print(f"{p_value} bu say?? {p_ths}'den b??y??kt??r, H0 reddedilemez. ??ki data da ayn?? da????l??mdad??r")
        else:
            print(f"{p_value} bu say?? {p_ths}'den k??????kt??r, H0 red edilir. ??ki Data aras??nda da????l??m fark?? vard??r")
    elif (p_value1 > p_ths and p_value2 < p_ths) or (p_value1 < p_ths and p_value2 > p_ths) or (p_value1 < p_ths and p_value2 < p_ths):
        t_stat, p_value = mannwhitneyu(data1[degisken], data2[degisken])

        if p_value > p_ths:
            print(f"{p_value} bu say?? {p_ths}'den b??y??kt??r, H0 reddedilemez. ??ki data da ayn?? da????l??mdad??r")
        else:
            print(f"{p_value} bu say?? {p_ths}'den k??????kt??r, H0 red edilir. ??ki Data aras??nda da????l??m fark?? vard??r")



def feature_engineering_titanic(df):

    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    # Name count
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    # name word count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    # family size
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    # age_pclass
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    # is alone
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level

    df.loc[(df['AGE'] < 8), 'NEW_AGE_CAT'] = 'child'
    df.loc[(df['AGE'] >= 8) & (df['AGE'] < 18), 'NEW_AGE_CAT'] = 'student'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 30), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 30) & (df['AGE'] < 60), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 60), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
    return df


def numeric_ozet(data, degisken, plot=False, null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan numeric de??i??kenlerin de??i??ken tiplerinin say??s??n?? ve totale kar???? oran??n?? bulur.
    Ayr??ca iste??e ba??l?? olarak de??i??ken da????l??m??n??n grafi??ini ve de??i??ken i??inde bulunan null say??s??n?? ????kart??r.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric de??i??kenin bulundu??u dataset.
    degisken:String
    Categoric de??i??ken ismi.
    plot:bool
    Fonksiyonda categoric de??i??ken da????l??m??n??n grafi??ini ??izdirmek i??in opsiyonel ??zellik.
    null_control:bool
    Fonksiyonda de??i??ken i??inde null de??er kontol?? i??in opsiyonel ??zellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique de??i??kenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(data[degisken].describe(quantiles).T)

    if plot:
        data[degisken].hist(bins=20)
        plt.xlabel(degisken)
        plt.title(degisken)
        plt.show(block=True)
    print("##########################################")

    if null_control:
        print(f"Null veri say??s??: {data[degisken].isnull().sum()}")


def target_analyser(dataframe, target, num_deg, cat_deg):
    for degisken in dataframe.columns:
        if degisken in cat_deg:
            print(degisken, ":", len(dataframe[degisken].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[degisken].value_counts(),
                                "RATIO": dataframe[degisken].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(degisken)[target].mean()}), end="\n\n\n")
        if degisken in num_deg:
            print(pd.DataFrame({
                "TARGET_MEAN": dataframe.groupby(target)[degisken].mean()}), end="\n\n\n")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def model_karsilastirma(df,model,target):
    X = df.drop(columns=target)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.15,
                                                        random_state=42)
    model_fit=model().fit(X_train,y_train)
    y_pred = model_fit.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(model ,"i??in sonu?? do??ruluk de??eri:",acc)
    return acc


def diabet_feature_engineering(data, target):
    # G??REV1
    # Ad??m1: Datan??n genel resmini inceleyin
    dataset_ozet(data)

    # Ad??m2:Numerik ve Kategorik de??i??kenleri yakalay??n
    cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(data, 10, 20)

    # Ad??m3: Kategorik ve Numerik de??i??kenleri incele
    for i in cat_deg:
        categoric_ozet(data, i,True,True)

    for i in num_deg:
        numeric_ozet(data, i,True,True)

    # Ad??m4:Hedef de??i??ken analizi yap??n??z. (Kategorik de??i??kenlere g??re hedef de??i??kenin ortalamas??, hedef de??i??kene g??re numerik de??i??kenlerin ortalamas??)

    target_analyser(data, target, num_deg, cat_deg)

    # Ad??m5: Ayk??r?? g??zlem analizi yap??n??z.
    for i in num_deg:
        s = check_outlier(data, i)
        print(f"{i} i??in outlier de??er var m??: {s}")

    # Ad??m6: Eksik g??zlem analizi yap??n??z.
    data.isnull().sum()

    # Ad??m7: Korelasyon analizi yap??n??z.
    data.corr(method="pearson")

    # G??REV2

    # Ad??m1:Ad??m 1: Eksik ve ayk??r?? de??erler i??in gerekli i??lemleri yap??n??z. Veri setinde eksik g??zlem bulunmamakta ama Glikoz, Insulin vb.
    # de??i??kenlerde 0 de??eri i??eren g??zlem birimleri eksik de??eri ifade ediyor olabilir. ??rne??in; bir ki??inin glikoz veya insulin de??eri 0
    # olamayacakt??r. Bu durumu dikkate alarak s??f??r de??erlerini ilgili de??erlerde NaN olarak atama yap??p sonras??nda eksik
    # de??erlere i??lemleri uygulayabilirsiniz.

    data["Insulin"].replace({0: np.nan}, inplace=True)
    data["Glucose"].replace({0: np.nan}, inplace=True)
    data["BloodPressure"].replace({0: np.nan}, inplace=True)
    data["Age"].replace({0: np.nan}, inplace=True)
    data["BMI"].replace({0: np.nan}, inplace=True)
    data["SkinThickness"].replace({0: np.nan}, inplace=True)

    for i in num_deg:
        data = threshold_degisimi(data, i)

    null_cols = missing_values_table(data, True)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data.loc[(data["Age"] < 8), 'NEW_AGE_CAT'] = 'child'
    data.loc[(data["Age"] >= 8) & (data["Age"] < 18), 'NEW_AGE_CAT'] = 'student'
    data.loc[(data["Age"] >= 18) & (data["Age"] < 30), 'NEW_AGE_CAT'] = 'young'
    data.loc[(data["Age"] >= 30) & (data["Age"] < 60), 'NEW_AGE_CAT'] = 'mature'
    data.loc[(data["Age"] >= 60), 'NEW_AGE_CAT'] = 'senior'
    data.groupby("NEW_AGE_CAT")["Outcome"].mean()

    for i in null_cols:
        data[i].fillna(data.groupby("NEW_AGE_CAT")[i].transform("mean"), inplace=True)

    # Ad??m 2: Yeni de??i??kenler olu??turunuz.

    data.loc[(data["BMI"] < 18.5), 'NEW_BMI_CAT'] = 'Under Weight'
    data.loc[(data["BMI"] >= 18.5) & (data["BMI"] < 25), 'NEW_BMI_CAT'] = 'Healthy Weight'
    data.loc[(data["BMI"] >= 25) & (data["BMI"] < 30), 'NEW_BMI_CAT'] = 'Over Weight'
    data.loc[(data["BMI"] >= 30), 'NEW_BMI_CAT'] = 'Obese'
    data.groupby("NEW_BMI_CAT")["Outcome"].mean()

    data.loc[(data["BloodPressure"] < 60), 'NEW_PRSR_CAT'] = 'Low Pressure'
    data.loc[(data["BloodPressure"] >= 60) & (data["BloodPressure"] < 80), 'NEW_PRSR_CAT'] = 'Ideal Pressure'
    data.loc[(data["BloodPressure"] >= 80) & (data["BloodPressure"] < 90), 'NEW_PRSR_CAT'] = 'Pre-high Pressure'
    data.loc[(data["BloodPressure"] >= 90), 'NEW_PRSR_CAT'] = 'High Pressure'
    data.groupby("NEW_PRSR_CAT")["Outcome"].mean()

    data.loc[(data["Pregnancies"] < 1), 'NEW_MOM_CAT'] = 'Not Mom'
    data.loc[(data["Pregnancies"] >= 1), 'NEW_MOM_CAT'] = 'Mom'
    data.groupby("NEW_MOM_CAT")["Outcome"].mean()
    # Bu de??i??ken ??ok manal?? olmad?? ama yinede tutaca????m.

    # Ad??m 3: Encoding i??lemlerini ger??ekle??tiriniz.
    cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(data, 10, 20)
    cat_deg = [i for i in cat_deg if i != "Outcome"]
    # ohe_cols = [col for col in cat_deg if 10 >= data[col].nunique() > 2] Bu yolla 2 ed??i??kenli olan se??enek label encodingde yap??labilirdi.
    data = one_hot_encoder(data, cat_deg)

    # Ad??m 4: Numerik de??i??kenler i??in standartla??t??rma yap??n??z.
    scaler = StandardScaler()
    data[num_deg] = scaler.fit_transform(data[num_deg])

    # Ad??m 5: Model olu??turunuz.
    for mod in models:
        model_karsilastirma(data, mod,target)

def telco_feature_engineering(data,target):
    data["Churn"].replace({"Yes": 1, "No": 0}, inplace=True)
    data["TotalCharges"].replace({" ": 0, }, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], downcast="float")

    data["TotalCharges"].replace({0: np.nan}, inplace=True)

    # G??REV1
    # Ad??m1: Datan??n genel resmini inceleyin
    dataset_ozet(data)
    # Ad??m2:Numerik ve Kategorik de??i??kenleri yakalay??n
    cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(data, 10, 20)
    # Ad??m3: Kategorik ve Numerik de??i??kenleri incele
    for i in cat_deg:
        categoric_ozet(data, i, True, True)

    for i in num_deg:
        numeric_ozet(data, i, True, True)

    # Ad??m4:Hedef de??i??ken analizi yap??n??z. (Kategorik de??i??kenlere g??re hedef de??i??kenin ortalamas??,
    # hedef de??i??kene g??re numerik de??i??kenlerin ortalamas??)
    target_analyser(data, target, num_deg, cat_deg)
    # Ad??m5: Ayk??r?? g??zlem analizi yap??n??z.
    for i in num_deg:
        s = check_outlier(data, i)
        print(f"{i} i??in outlier de??er var m??: {s}")

    # Ad??m6: Eksik g??zlem analizi yap??n??z.
    print(data.isnull().sum())
    # Ad??m7: Korelasyon analizi yap??n??z.
    data.corr(method="pearson")
    # G??REV2

    # Ad??m1:Ad??m 1: Eksik ve ayk??r?? de??erler i??in gerekli i??lemleri yap??n??z. Veri setinde eksik g??zlem bulunmamakta ama Glikoz, Insulin vb.
    # de??i??kenlerde 0 de??eri i??eren g??zlem birimleri eksik de??eri ifade ediyor olabilir. ??rne??in; bir ki??inin glikoz veya insulin de??eri 0
    # olamayacakt??r. Bu durumu dikkate alarak s??f??r de??erlerini ilgili de??erlerde NaN olarak atama yap??p sonras??nda eksik
    # de??erlere i??lemleri uygulayabilirsiniz.

    print(data.isnull().sum())

    for i in num_deg:
        data = threshold_degisimi(data, i)

    null_cols = missing_values_table(data, True)
    print(null_cols)

    for i in null_cols:
        data[i].fillna(data.groupby("gender")[i].transform("mean"), inplace=True)

    print(data.isnull().sum())
    # Ad??m 2: Yeni de??i??kenler olu??turunuz.
    data.tenure.describe()
    data.loc[(data["tenure"] < 10), 'NEW_TENURE'] = 'New Customer'
    data.loc[(data["tenure"] >= 10) & (data["tenure"] < 25), 'NEW_TENURE'] = 'Ptentials'
    data.loc[(data["tenure"] >= 25) & (data["tenure"] < 50), 'NEW_TENURE'] = 'Loyals'
    data.loc[(data["tenure"] >= 50), 'NEW_TENURE'] = 'Champs'
    data.groupby('NEW_TENURE')["Churn"].mean()

    data["New_Total_Income"] = data["tenure"] * data["MonthlyCharges"]
    data.MonthlyCharges.describe()
    data.loc[(data["MonthlyCharges"] < 40), 'NEW_MNT_INCOME_CAT'] = 'Low'
    data.loc[(data["MonthlyCharges"] >= 40) & (data["MonthlyCharges"] < 70), 'NEW_MNT_INCOME_CAT'] = 'Medium'
    data.loc[(data["MonthlyCharges"] >= 70) & (data["MonthlyCharges"] < 90), 'NEW_MNT_INCOME_CAT'] = 'Ideal'
    data.loc[(data["MonthlyCharges"] >= 90), 'NEW_MNT_INCOME_CAT'] = 'High'
    data.groupby("NEW_MNT_INCOME_CAT")["Churn"].mean()

    # Ad??m 3: Encoding i??lemlerini ger??ekle??tiriniz.
    cat_deg, num_deg, car_deg = degisken_tiplerine_ayirma(data, 10, 20)
    cat_deg = [i for i in cat_deg if i != "Churn"]
    data = one_hot_encoder(data, cat_deg)
    # Ad??m 4: Numerik de??i??kenlerin Scaling i??lemlerini yap??n
    scaler = StandardScaler()
    data[num_deg] = scaler.fit_transform(data[num_deg])

    data.drop(columns="customerID", inplace=True)
    # Ad??m 5: Model olu??turunuz.
    for mod in models:
        model_karsilastirma(data, mod, target)