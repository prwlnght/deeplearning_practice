'''
The purpose of this to visualize some of this data and do some analyses

copyright @prwl_nght

'''

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

data_folder = os.path.join('tmp', 'Features')

counter = 0
for m_folder in os.listdir(data_folder):
    if not (m_folder.startswith('.') or m_folder.startswith('desktop')):
        counter += 1
        print(m_folder)
        m_file = os.path.join(data_folder, m_folder, 'features.csv')  # assumign this
        df2 = pd.read_csv(m_file)

        features = list(df2)[2:]
        # filter by mean only
        features = list(itertools.compress(features, ['mean' in x for x in features]))
        # filter by signal
        features = list(itertools.compress(features, ['ORN' in x or 'GYR' in x for x in features]))

        x = df2.loc[:, features].values

        y = df2.loc[:, ['Name']].values

        x = StandardScaler().fit_transform(x)

        pca_df = pd.DataFrame(data=x, columns=features).head()

        pca = PCA(n_components=2)

        principalComponents = pca.fit_transform(x)
        pca_to_show = pd.DataFrame(pca.components_, columns=features, index=['PC-1', 'PC-2'])

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, df2[['Name']]], axis=1)
        finalDf.head(5)
        explained_variance = pca.explained_variance_ratio_
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('Principal Component 1: {0:.2f}'.format(explained_variance[0]), fontsize=15)
        ax.set_ylabel('Principal Component 2: {0:.2f}'.format(explained_variance[1]), fontsize=15)
        ax.set_title('2 Comp. PCA for user_' + str(counter), fontsize=20)

        targets = ['alphabet_b', 'alphabet_h', 'alphabet_j', 'alphabet_m', 'alphabet_n', 'alphabet_q']
        colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['Name'] == target
            ax.scatter(
                finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c=color
                , s=50)
        ax.legend(targets)
        ax.grid()
        filename = 'TwoCompPCA_features_User_' + str(counter)
        filename = os.path.join('tmp', 'figures_pca', filename)
        plt.savefig(filename)

        pca_to_show.to_csv(os.path.join('tmp', 'TwoCompPCA_features_User_' + str(counter) + '.csv'))
