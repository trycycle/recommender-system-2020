import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity


class CF:
    def __init__(self):
        pass

    
    def cosine(self, M):
        _M = M.copy()
        _M[np.isnan(_M)] = 0
        similarities = cosine_similarity(_M)
        return similarities
    
    
    def classify_k_nn(self, sim_vec, k=2):
        """ 類似度ベクトルsim_vecを用いて、類似度上位k件の要素をTrueとするリストを返す
        """
        # 類似度順位を返す。なお、順位は小さいもの順に割り振られる
        sim_rank = rankdata(sim_vec, method='ordinal')
        k_nn = list(pd.Series(sim_vec).rank(ascending=False) <= k)
        return k_nn


class UserBasedCF(CF):

    def predict_rating(self, df, target_user, target_item, sim_threshold=0.7):
        """ 評価値行列からtarget_userのtarget_itemに対する評価値を予想する
        """
        # 類似度行列
        sim_matrix = df.T.corr(method='pearson').values
        # ターゲットユーザと他ユーザとの類似度ベクトル
        sim_vec = sim_matrix[target_user, :]   
        # target_itemを評価していないユーザの類似度を無視する
        sim_vec[np.isnan(df.values[:, target_item])] = np.nan

        # target_userと他ユーザの類似度を取得し、
        # 類似度がthreshold以上のユーザを最近傍（Nearest Neighbor: nn）とする
        nn = (sim_vec >= sim_threshold)

        # 最近傍ユーザの類似度を取得
        k_nn_sim_vec = sim_matrix[target_user, :][nn]

        # 最近傍ユーザのtarget_itemに対する評価値を取得
        rating_matrix = df.values    
        rating_vec = rating_matrix[nn, target_item]

        # 平均評価値をユーザ毎に求める
        mean_vec = df.mean(axis=1).values

        predicted = np.dot(k_nn_sim_vec, (rating_vec - mean_vec[nn])) / sum(k_nn_sim_vec) + mean_vec[target_user]
        return predicted

   
    def predict_rating_with_k_nn(self, df, target_user, target_item, k=2):
        """ 評価値行列からtarget_userのtarget_itemに対する評価値を予想する
        """
        # ユーザ総数とアイテム総数
        user_num, item_num = df.shape

        # 類似度行列
        sim_matrix = df.T.corr(method='pearson').values
        # ターゲットユーザと他ユーザとの類似度ベクトル
        sim_vec = sim_matrix[target_user, :]
        # target_itemを評価していないユーザの類似度を無視する
        sim_vec[np.isnan(df.values[:, target_item])] = np.nan

        # target_userと類似するユーザの上位threshold件についてTrue、
        # それ以外をFalseとするリストを返す
        k_nn = self.classify_k_nn(sim_vec, k)
        # 最近傍ユーザの類似度を取得
        k_nn_sim_vec = sim_vec[k_nn]

        # 最近傍ユーザのtarget_itemに対する評価値を取得
        rating_matrix = df.values    
        rating_vec = rating_matrix[k_nn, target_item]

        # 平均評価値をユーザ毎に求める
        mean_vec = df.mean(axis=1).values

        predicted = np.dot(k_nn_sim_vec, (rating_vec - mean_vec[k_nn])) / sum(k_nn_sim_vec) + mean_vec[target_user]
        return predicted
    
    
    

class ItemBasedCF(CF):
    
    def predict_rating_with_k_nn(self, df, target_user, target_item, k=2):
        """ 評価値行列からtarget_userのtarget_itemに対する評価値を予想する
        """
        # ユーザ総数とアイテム総数
        user_num, item_num = df.shape

        # ユーザごとの評価値平均ベクトル
        mean_vec = np.reshape(df.mean(axis=1).values, (user_num, 1))
        mod_rating_matrix = df.values - mean_vec # 行がアイテム，列がユーザに対応

        # 類似度の取得
        sim_matrix = self.cosine(mod_rating_matrix.T)
        sim_vec = sim_matrix[target_item, :]
        # target_userが評価していないアイテムの類似度を無視する
        sim_vec[np.isnan(mod_rating_matrix[target_user, :])] = np.nan

        # target_itemと類似するアイテムの上位threshold件についてTrue、
        # それ以外をFalseとするリストを返す
        k_nn = self.classify_k_nn(sim_vec, k)
        # 最近傍ユーザの類似度を取得
        k_nn_sim_vec = sim_vec[k_nn]

        # 最近傍アイテムに対するtarget_userの評価値を取得
        rating_vec = mod_rating_matrix[target_user, k_nn]

        # 評価値の計算
        predicted = mean_vec[target_user, 0] + np.dot(k_nn_sim_vec, rating_vec) / sum(k_nn_sim_vec)
        return predicted