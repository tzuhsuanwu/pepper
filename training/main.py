from pathlib import Path
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


train_csv = Path('./data/train.csv')
test_csv = Path('./data/test.csv')

wt_columns = [
    "wt_fa_atr", "wt_fa_rep", "wt_fa_intra_rep", "wt_fa_sol", "wt_lk_ball_wtd", "wt_fa_intra_sol_xover4",
    "wt_fa_elec", "wt_hbond_lr_bb", "wt_hbond_sr_bb", "wt_hbond_bb_sc", "wt_hbond_sc", "wt_dslf_fa13",
    "wt_rama_prepro", "wt_p_aa_pp", "wt_fa_dun", "wt_omega", "wt_pro_close", "wt_yhh_planarity", "wt_ref", "wt_sum"
]

mu_columns = [
    "mu_fa_atr", "mu_fa_rep", "mu_fa_intra_rep", "mu_fa_sol", "mu_lk_ball_wtd", "mu_fa_intra_sol_xover4",
    "mu_fa_elec", "mu_hbond_lr_bb", "mu_hbond_sr_bb", "mu_hbond_bb_sc", "mu_hbond_sc", "mu_dslf_fa13",
    "mu_rama_prepro", "mu_p_aa_pp", "mu_fa_dun", "mu_omega", "mu_pro_close", "mu_yhh_planarity", "mu_ref", "mu_sum"
]

diff_columns = [
    "diff_fa_atr", "diff_fa_rep", "diff_fa_intra_rep", "diff_fa_sol", "diff_lk_ball_wtd", "diff_fa_intra_sol_xover4",
    "diff_fa_elec", "diff_hbond_lr_bb", "diff_hbond_sr_bb", "diff_hbond_bb_sc", "diff_hbond_sc", "diff_dslf_fa13",
    "diff_rama_prepro", "diff_p_aa_pp", "diff_fa_dun", "diff_omega", "diff_pro_close", "diff_yhh_planarity", "diff_ref", "diff_sum"
]

whole_column = wt_columns + mu_columns + diff_columns


def training():
    # Load csv
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Select features in dataframe
    X_train = train_df[whole_column + ['wt_SASA', 'mu_SASA', 'diff_SASA']]
    y_train = train_df["true_label"]
    X_test = test_df[whole_column + ['wt_SASA', 'mu_SASA', 'diff_SASA']]
    y_test = test_df["true_label"]

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=1500,
        learning_rate=0.1,
        num_leaves=27,
        scale_pos_weight=0.364,
    )

    # Feature selection
    selector = VarianceThreshold(0.01)
    selector = selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    # Train
    clf.fit(X_train, y_train)

    # Save scaler, selector and classifier
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('selector.pkl', 'wb') as f:
        pickle.dump(selector, f)

    with open('lgbm_prediction_model.pkl', 'wb') as f:
        pickle.dump(clf, f)


def main():
    training()


if __name__ == "__main__":
    main()
