#import seaborn as sns; sns.set(font_scale=1.2)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylustrator

np.random.seed(1)
pylustrator.start()
def smooth_data(window_size, data):
    #np_array input
    y = np.ones(window_size)
    for idx in range(data.shape[0]):
        x = data[idx,:]
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        data[idx,:] = smoothed_x
    return data

def get_array(name):
    epoches = 30000
    dtpc_r_all = []
    # DTPC
    with open(name, 'r') as f:
        f.readline()
        for i in range(epoches):
            tmp = f.readline().split(',')
            dtpc_r_all.append(eval(tmp[-1]) / 100.0 )
    dtpc_r_all = np.array(dtpc_r_all).reshape(300, 100).T
    dtpc_r_all_idx = np.random.permutation(100)
    dtpc_r_all = dtpc_r_all[dtpc_r_all_idx[:10],:]
    dtpc_r_all = smooth_data(50,dtpc_r_all)

    return dtpc_r_all


def get_data():
    '''获取数据
    '''
    top_path = r'E:\GitCode\ICC_2022_CBQ\models\uav_com'
    basecond = get_array(top_path+r'\DTPC\run1\logs\all_reward.csv')
    cond1 = get_array(top_path+r'\maddpg\run1\logs\all_reward.csv')
    cond2 = get_array(top_path+r'\ddpg\run1\logs\all_reward.csv')
    cond3 = get_array(top_path+r'\dqn\run1\logs\all_reward.csv')
    cond4 = get_array(top_path+r'\double_dqn\run1\logs\all_reward.csv')
    cond5 = get_array(top_path+r'\dueling_dqn\run1\logs\all_reward.csv')

    return cond1, basecond,  cond2, cond3, cond4, cond5

data = get_data()
label = ['MADDPG', 'DTPC',  'DDPG-[13]', 'DQN','Double DQN-[12]', 'Dueling DQN-[10]' ]
df=[]
for i in range(len(data)):
    tmp = pd.DataFrame(data[i]).melt(var_name='Episode', value_name='Reward')
    tmp.Episode *= 100
    df.append(tmp)
    df[i]['Algos']= label[i]


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
t = np.linspace(0, 299, 300)
color_list  =['b', #blue
                'g', #green
                'r', #red
                'c', #cyan
                'm', #magenta
                'y', #yellow
                'k', #black
                'w' #white
                ]
for i,label_txt in enumerate(label):
    ax1.plot(t, data[i][0],  label=label_txt, color = color_list[i])
ax1.legend(loc="best")
ax1.grid()
ax1.set_facecolor("white")
ax1.set_ylabel('Reward',fontsize='large')
ax1.set_xlabel('Epoch*100', fontsize='large')
fig.patch.set_facecolor('white')
#ax1.grid(b=True, which='major', color='tab:gray', linestyle='-')
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].get_legend()._set_loc((0.609487, 0.048551))
#% end: automatic generated code from pylustrator
plt.show()
plt.savefig('learning_curve.pdf',dpi = 400)

