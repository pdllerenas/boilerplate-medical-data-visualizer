import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
# 2
df['overweight'] = np.where((df['weight']/((df['height']/100)**2)) > 25, 1, 0)

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0 
df.loc[df['cholesterol'] > 1, 'cholesterol']= 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], value_name='value', var_name='variable')

    # 6
    df_cat = df_cat.groupby('cardio', as_index=False).value_counts()
    df_cat = df_cat.rename(columns={'count': 'total'})

    # 7
    df_cat = df_cat.astype({'cardio': 'float', 'value': 'float', 'total': 'float'})


    # 8
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    fig = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio", order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']).figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))] 

    # 12
    corr = df_heat.corr()
    print(corr)
    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots()

    # 15
    fig = sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=.5, ax=ax).figure


    # 16
    fig.savefig('heatmap.png')
    return fig
