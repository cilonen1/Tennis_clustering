import pandas as pd  # start seas 18-19
import numpy as np
from sklearn import tree
from sklearn.cluster import KMeans


def count_sum(continent, ctry_border, feat, cols, year, kmeans, cl_num):
    st, end = ctry_border[continent]
    df_cont = pd.read_excel(
        r"C:\Users\cilon\OneDrive\Desktop\New-Tennis\Chal_full_by_cont.xlsx",
        header=0,
        sheet_name=continent,
    )[st:end][cols]
    df_cont.fillna({"Rank2": 1200, "Rank1": 1200}, inplace=True)
    df_cont.fillna(0.5, inplace=True)
    df_cont["Date"] = pd.to_datetime(df_cont.Date, dayfirst=True)
    df_cont_cluster = df_cont[df_cont.Rd.isin(["F", "SF", "QF"])].copy()
    X_tr = df_cont_cluster[feat]
    df_cont_cluster["label"] = kmeans.predict(X_tr)
    label_feat = ["Total rate", "2m point"]
    df_final = pd.DataFrame()
    b_date = pd.to_datetime(f"{year}-01-01", dayfirst=True)
    y_test = []

    for i in range(cl_num):
        df_label = df_cont_cluster[df_cont_cluster.label == i].copy()
        X_train = df_label[df_label.Date < b_date][label_feat]
        y_train = df_label[df_label.Date < b_date]["win"]
        regDT = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=40)
        regDT.fit(X_train, y_train)
        X_test = df_label[df_label.Date > b_date][label_feat]
        y_test.extend(regDT.predict(X_test))
        df_final = df_final.append(df_label[df_label.Date > b_date])

        df_final["predict"] = y_test
        df_final["bet1"] = np.where(df_final.predict * df_final.k1 > 1, 1, 0)
        df_final["bet2"] = np.where((1 - df_final.predict) * df_final.k2 > 1, 1, 0)
        df_final["money1"] = np.where(
            df_final.win == 1,
            100 * df_final.bet1,
            (-100 / (df_final.k1 - 1) * df_final.bet1).astype(int),
        )
        df_final["money2"] = np.where(
            df_final.win == 0,
            100 * df_final.bet2,
            (-100 / (df_final.k2 - 1) * df_final.bet2).astype(int),
        )
    bet_sum = df_final[["money1", "money2"]].sum()
    print(bet_sum)
    return df_final


def full_cycle(continent, year=2019):
    print("process!")
    ctry_border = {
        "WE": [2790, 8624],
        "US": [7779, 1340],
        "S.Am": [3331, 6702],
        "EastEu": [3875, 7344],
        "WE_Clay": [3071, 12040],
    }
    cols = [
        "Date",
        "Loc",
        "Rd",
        "PL1",
        "PL2",
        "k1",
        "k2",
        "П1",
        "П2",
        "Total rate",
        "Total1",
        "Total2",
        "Games1",
        "Games2",
        "Poi_s1",
        "Poi_s2",
        "Poi_t1",
        "Poi_t2",
        "Surf p",
        "Surf1",
        "Surf2",
        "2m point",
        "Diff",
        "Rank1",
        "Rank2",
        "win",
        "loss",
    ]
    dx = pd.read_excel(
        r"C:\Users\cilon\OneDrive\Desktop\New-Tennis\CHAL16-21(F).xlsx",
        header=0,
        sheet_name="Archive",
    )[:26597][cols]
    dx.fillna(
        {"Rank2": 1200, "Rank1": 1200}, inplace=True
    )  # берем все челленджеры чтобы из них один раз сделать кластеры
    dx.fillna(0.5, inplace=True)
    dx["Date"] = pd.to_datetime(dx.Date, dayfirst=True)
    feat = ["Poi_s1", "Poi_s2", "2m point"]
    X_cluster = dx[dx.Rd.isin(["F", "SF", "QF"])][feat]

    cl_num = 9
    kmeans = KMeans(n_clusters=cl_num, random_state=0)
    kmeans.fit(X_cluster)  # создали кластеры

    df_final = count_sum(continent, ctry_border, feat, cols, year, kmeans, cl_num)
    return df_final
