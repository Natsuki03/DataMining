import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

#ex1.1

def true_function(x):
    '''
    >>> true_function(0) == 0
    True
    '''
    y = np.sin(np.pi * x * 0.8) * 10
    return y

"""
x = np.arange(-1,1,0.01)

plt.plot(x,true_function(x),label="y = $\sin(\pi * x * 0.8) * 10$")
plt.title("ture_function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(path + "ex1.1.png")
plt.show()
"""

"""
#ex1.2
random.seed(0)

n = [round(random.uniform(-1, 1), 2) for _ in range(20)]
true_val = [true_function(i) for i in n]

data = np.array([n, true_val]).T

df = pd.DataFrame(data, columns=["観測点", "真値"])


for i in range(len(n)):
    plt.plot(n[i], true_val[i], marker=".", markersize=10, color="red")
plt.savefig(path + "ex1.2.png")
plt.show()
"""

"""
#ex1.3
noize = [round(random.normalvariate(mu=0.0, sigma=2.0)/2, 2) for _ in range(20)]

noize_data = df["真値"] + noize
df["観測値"] = np.array(noize_data).T


for i in range(len(noize)):
    plt.plot(n[i], noize_data[i], marker="^", markersize=5, color="green")
plt.savefig(path + "ex1.3.png")
plt.show()
"""

#ex1.4
#df.to_csv(path + "dataset1.tsv", sep="\t", index=False)

#ex1.5
#dataset = pd.read_csv(path + "dataset1.tsv", sep="\t")

#ex1.6

def sample_plot(x_min, x_max, step, df, path): #plotする関数
    x = np.arange(x_min, x_max, step)

    plt.plot(x,true_function(x),label="True_function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.savefig(path + "ex1.1.png")

    plt.scatter(df["観測点"].values, df["真値"].values, marker=".", color="red", label="True_data")
    plt.legend(loc="best")
    plt.savefig(path + "ex1.2.png")

    plt.scatter(df["観測点"].values, df["観測値"].values, marker="^", color="green", label="観測値")
    plt.legend(loc="best")
    plt.savefig(path + "ex1.3.png")

def make_true_data(data_size, min_val, max_val, column_name, random_state=0):
    random.seed(random_state)
    n = [round(random.uniform(min_val, max_val), 2) for _ in range(data_size)]
    true_val = [true_function(i) for i in n]

    data = np.array([n, true_val]).T

    df = pd.DataFrame(data, columns=column_name)
    return df

def make_noize(data_size, average, std, df):
    noize = [round(random.normalvariate(average, std)/2, 2) for _ in range(data_size)]

    noize_data = df["真値"] + noize
    df["観測値"] = np.array(noize_data).T
    return df

def save_df_to_tsv(df, path, filename):
    df.to_csv(path + filename, sep="\t", index=False)

def load_tsv_from_df(path, filename):
    dataset = pd.read_csv(path + filename, sep="\t")
    return dataset

if __name__ == '__main__':
    path = "/Users/e195767/VS code/Data_Mining_Code/exam/"
    x_min = -1
    x_max = 1
    step = 0.01
    df = make_true_data(data_size=20, min_val=-1, max_val=1, column_name=["観測点", "真値"]) #ノイズなし
    df = make_noize(data_size=20, average=0.0, std=2.0, df=df) #ノイズ付与
    sample_plot(x_min, x_max, step, df, path) #plot
    save_df_to_tsv(df, path, filename="dataset1.tsv") #tsvファイル形式で保存
    dataset = load_tsv_from_df(path, filename="dataset1.tsv") #tsvファイルの読み込み