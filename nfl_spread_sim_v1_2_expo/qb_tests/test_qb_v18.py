import pandas as pd
from score_model.qb_tuned_v18 import shrink_qb_features

def test_shrinkage_reduces_extremes():
    df=pd.DataFrame([{
        "home_qb_prior_dropbacks":10,
        "away_qb_prior_dropbacks":10,
        "home_pre_qb_epa_per_dropback":1.0,
        "away_pre_qb_epa_per_dropback":-1.0,
        "home_pre_qb_success_rate":.7,"away_pre_qb_success_rate":.3,
        "home_pre_qb_cpoe":10.,"away_pre_qb_cpoe":-10.,
        "home_pre_qb_sack_rate":.01,"away_pre_qb_sack_rate":.2,
        "home_pre_qb_interception_rate":.01,"away_pre_qb_interception_rate":.1,
        "home_pre_qb_explosive_pass_rate":.3,"away_pre_qb_explosive_pass_rate":.05,
    },{
        "home_qb_prior_dropbacks":1000,
        "away_qb_prior_dropbacks":1000,
        "home_pre_qb_epa_per_dropback":.5,
        "away_pre_qb_epa_per_dropback":-.5,
        "home_pre_qb_success_rate":.6,"away_pre_qb_success_rate":.4,
        "home_pre_qb_cpoe":5.,"away_pre_qb_cpoe":-5.,
        "home_pre_qb_sack_rate":.05,"away_pre_qb_sack_rate":.1,
        "home_pre_qb_interception_rate":.02,"away_pre_qb_interception_rate":.05,
        "home_pre_qb_explosive_pass_rate":.2,"away_pre_qb_explosive_pass_rate":.1,
    }])
    out=shrink_qb_features(df,300)
    assert abs(out.loc[0,"diff_qb_epa_per_dropback"]) < 2.0
