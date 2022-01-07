import pandas as pd
import numpy as np
import Utils_cagri as util
diabet=pd.read_csv("diabetes.csv")
telco=pd.read_csv("Telco-Customer-Churn.csv")


while True:
    giris = input("Diabet dataseti özellik mühendisliği için 1 \nTelco dataseti özellik ühendisi için 2 yi tuşlayın")
    if giris=="1":
        util.diabet_feature_engineering(diabet,"Outcome")
        break
    elif giris=="2":
        util.telco_feature_engineering(telco,"Churn")
        break
    else:
        print("Hatalı giriş yaptınız.Lütfen tekrar deneyiniz")