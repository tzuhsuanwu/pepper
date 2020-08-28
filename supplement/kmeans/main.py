from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

wt_columns = ["wt_fa_atr", "wt_fa_rep", "wt_fa_intra_rep", "wt_fa_sol", "wt_lk_ball_wtd", "wt_fa_intra_sol_xover4",
              "wt_fa_elec", "wt_hbond_lr_bb", "wt_hbond_sr_bb", "wt_hbond_bb_sc", "wt_hbond_sc", "wt_dslf_fa13",
              "wt_rama_prepro", "wt_p_aa_pp", "wt_fa_dun", "wt_omega", "wt_pro_close", "wt_yhh_planarity", "wt_ref", "wt_sum"]

mu_columns = ["mu_fa_atr", "mu_fa_rep", "mu_fa_intra_rep", "mu_fa_sol", "mu_lk_ball_wtd", "mu_fa_intra_sol_xover4",
              "mu_fa_elec", "mu_hbond_lr_bb", "mu_hbond_sr_bb", "mu_hbond_bb_sc", "mu_hbond_sc", "mu_dslf_fa13",
              "mu_rama_prepro", "mu_p_aa_pp", "mu_fa_dun", "mu_omega", "mu_pro_close", "mu_yhh_planarity", "mu_ref", "mu_sum"]

diff_columns = ["diff_fa_atr", "diff_fa_rep", "diff_fa_intra_rep", "diff_fa_sol", "diff_lk_ball_wtd", "diff_fa_intra_sol_xover4",
                "diff_fa_elec", "diff_hbond_lr_bb", "diff_hbond_sr_bb", "diff_hbond_bb_sc", "diff_hbond_sc", "diff_dslf_fa13",
                "diff_rama_prepro", "diff_p_aa_pp", "diff_fa_dun", "diff_omega", "diff_pro_close", "diff_yhh_planarity", "diff_ref", "diff_sum"]


def clustering(df):
    index = df["index"].to_list()
    print(df)
    df = df[wt_columns + mu_columns + diff_columns]
    kmeans = KMeans(n_clusters=1250, random_state=0)
    kmeans = kmeans.fit(df)
    means_index = pd.DataFrame(kmeans.transform(df)).idxmin(axis=0).to_list()
    return [index[i] for i in means_index]


def main():
    df = pd.read_csv("result.csv")
    df = df[df["predict_status"] == "success"]
    df = df.reset_index(drop=True)
    df = df.reset_index()
    p_df = df[df["true_label"] == 1]
    b_df = df[df["true_label"] == 0]
    p_index = clustering(p_df)
    b_index = clustering(b_df)
    test_index = p_index + b_index
    test_index.sort()
    test_df = df.loc[test_index]
    train_df = df.mask(df["SAV"].isin(test_df["SAV"])).dropna()

    train_df.to_csv("train_kmeans.csv", index=False)
    test_df.to_csv("test_kmeans.csv", index=False)

    train_sav = train_df["SAV"].to_list()
    with open("train_sav.txt", "w") as f:
        f.write("\n".join(train_sav))

    test_sav = test_df["SAV"].to_list()
    with open("test_sav.txt", "w") as f:
        f.write("\n".join(test_sav))


if __name__ == "__main__":
    main()
