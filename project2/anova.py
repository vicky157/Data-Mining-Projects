import pandas as pd
import scipy.stats as stats
print("For cho.txt:")
data = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'DBSCAN', 'HAC', 'HAC', 'GMM', 'GMM'],
    'Dataset': ['cho.txt', 'cho.txt', 'cho.txt', 'cho.txt', 'cho.txt', 'cho.txt'],
    'SilhouetteScore': [0.04, 0.05, 0.20, 0.21, 0.22, 0.23],
    'ARI': [0.04, 0.05, 0.41, 0.42, 0.36, 0.37]
})

f_value_silhouette, p_value_silhouette = stats.f_oneway(data[data['Algorithm'] == 'DBSCAN']['SilhouetteScore'],
                                                        data[data['Algorithm'] == 'HAC']['SilhouetteScore'],
                                                        data[data['Algorithm'] == 'GMM']['SilhouetteScore'])

f_value_ari, p_value_ari = stats.f_oneway(data[data['Algorithm'] == 'DBSCAN']['ARI'],
                                          data[data['Algorithm'] == 'HAC']['ARI'],
                                          data[data['Algorithm'] == 'GMM']['ARI'])

print(f"Silhouette Score: F-Value = {f_value_silhouette}, P-Value = {p_value_silhouette}")
print(f"ARI: F-Value = {f_value_ari}, P-Value = {p_value_ari}")


print('For Iyer.txt:')
import pandas as pd
import scipy.stats as stats

data_iyer = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'DBSCAN', 'HAC (Ward)', 'HAC (Ward)', 'HAC (Complete)', 'HAC (Complete)', 'GMM', 'GMM'],
    'Dataset': ['iyer.txt', 'iyer.txt', 'iyer.txt', 'iyer.txt', 'iyer.txt', 'iyer.txt', 'iyer.txt', 'iyer.txt'],
    'SilhouetteScore': [0.70, 0.71, 0.73, 0.74, 0.89, 0.90, 0.82, 0.83],
    'ARI': [0.03, 0.04, 0.02, 0.03, 0.00, 0.01, 0.003, 0.004]
})

f_value_silhouette_iyer, p_value_silhouette_iyer = stats.f_oneway(data_iyer[data_iyer['Algorithm'] == 'DBSCAN']['SilhouetteScore'],
                                                                  data_iyer[data_iyer['Algorithm'] == 'HAC (Ward)']['SilhouetteScore'],
                                                                  data_iyer[data_iyer['Algorithm'] == 'HAC (Complete)']['SilhouetteScore'],
                                                                  data_iyer[data_iyer['Algorithm'] == 'GMM']['SilhouetteScore'])

f_value_ari_iyer, p_value_ari_iyer = stats.f_oneway(data_iyer[data_iyer['Algorithm'] == 'DBSCAN']['ARI'],
                                                    data_iyer[data_iyer['Algorithm'] == 'HAC (Ward)']['ARI'],
                                                    data_iyer[data_iyer['Algorithm'] == 'HAC (Complete)']['ARI'],
                                                    data_iyer[data_iyer['Algorithm'] == 'GMM']['ARI'])

print(f"iyer.txt - Silhouette Score: F-Value = {f_value_silhouette_iyer}, P-Value = {p_value_silhouette_iyer}")
print(f"iyer.txt - ARI: F-Value = {f_value_ari_iyer}, P-Value = {p_value_ari_iyer}")
