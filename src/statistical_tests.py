import os
from csv import reader

import pandas as pd
from scipy import stats

from dunn import dunn

output_dir = 'stat_tests'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

result_directories = ['baseline_100', 'multi_evolution_100', 'single_evolution_100',
                      'baseline_20', 'multi_evolution_20', 'single_evolution_20']


def read_data():
    scores = {i: [] for i in result_directories}
    ranks = {i: [] for i in result_directories}

    for i in result_directories:
        with open(os.path.join(i, 'scores.csv'), 'r') as f:
            r = reader(f)
            for row in r:
                scores[i].extend([float(j) for j in row])

        with open(os.path.join(i, 'ranks.csv'), 'r') as f:
            r = reader(f)
            for row in r:
                ranks[i].extend([float(j) for j in row])

    return scores, ranks


def normality_test(data, filename):
    norm_test = {}
    for key in result_directories:
        norm_test[key] = list(stats.mstats.normaltest(data[key]))
    norm_test = pd.DataFrame(norm_test, columns=result_directories, index=['stat', 'p-val'])
    norm_test.to_csv(os.path.join(output_dir, filename))


def descriptive_statistics(data, filename):
    df = pd.DataFrame(data, columns=result_directories)
    df.describe().to_csv(os.path.join(output_dir, filename))


def kruskal_wallis_test(data, filename):
    res = {'100 group': None, '20 group': None}
    group1 = [data[i] for i in result_directories[:3]]
    group2 = [data[i] for i in result_directories[3:]]

    res['100 group'] = list(stats.kruskal(*group1))
    res['20 group'] = list(stats.kruskal(*group2))

    res = pd.DataFrame(res, index=['stat', 'p-val'])
    res.to_csv(os.path.join(output_dir, filename))


def dunn_test(data, filename):
    group1 = [data[i] for i in result_directories[:3]]
    group2 = [data[i] for i in result_directories[3:]]

    pd.DataFrame(dunn(*group1)).to_csv(os.path.join(output_dir, filename + '_100.csv'))
    pd.DataFrame(dunn(*group2)).to_csv(os.path.join(output_dir, filename + '_20.csv'))


scores, ranks = read_data()

normality_test(scores, 'score_normality_test.csv')
normality_test(ranks, 'rank_normality_test.csv')

descriptive_statistics(scores, 'score_stats.csv')
descriptive_statistics(ranks, 'rank_stats.csv')

kruskal_wallis_test(scores, 'score_kruskal_test.csv')
kruskal_wallis_test(ranks, 'rank_kruskal_test.csv')

dunn_test(scores, 'score_dunn_test')
dunn_test(ranks, 'rank_dunn_test')
