import matplotlib
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from odeintw import odeintw
import numpy as np
import pandas as pd


def f(v, s, k, l):
    '''
    수식
    '''
    # return (v[1], -(k**2)*v[0])
    return (v[1], -3*v[1] - (k**2)*v[0])


if __name__=="__main__":
    '''
    이 값을 바꿔서 real과 image값을 조절 (최소값, 최대값, 간격)
    k_r_index: real part
    k_i_index: image part
    '''
    k_r_index = np.linspace(0.001, 15, 999)
    k_i_index = [0.1, 1, 2]
    # k_i_index = np.linspace(0.1, 0.1, 1)

    result = pd.DataFrame(columns=['r_index', 'i_index', 'real_value', 'imag_value'])

    for i in tqdm(k_i_index):
        for r in tqdm(k_r_index, leave=False):
            v0 = [complex(0,0), complex(1,0)]
            ss = np.linspace(0, 1, 200)
            us, infodict = odeintw(f, v0, ss, args=(complex(r,i),0), full_output=True)
            vs_real = us[:, 0].real[-1]
            vs_imag = us[:, 0].imag[-1]

            temp = pd.DataFrame(data=[[r, i, vs_real, vs_imag]], columns=result.columns)
            result = pd.concat([result, temp], ignore_index=True)
            
            if abs(vs_real) < 1e-3:
                plt.plot(ss, us[:, 0].real, '-', label=f"{r:.2f}")
                plt.xlabel('s')
                plt.ylabel('V(s)')

        plt.legend()
        plt.savefig(f"data/{i}i.png")
        plt.clf()

    print(result)
    result.to_csv("data/result.csv")

    # plt.plot(ss,result["real_value"],'-')
    # plt.xlabel('s')
    # plt.ylabel('V(1)')
    # plt.savefig("data/mygraph_real.png")

    # plt.plot(result["i_index"],result["imag_value"],'-')
    # plt.xlabel('s')
    # plt.ylabel('V(1)')
    # plt.savefig("data/mygraph_imag.png")


    # plt.title(f'k = {r}+{j}'+'$\it{i}$')
    # i = complex(i, j)
    # if abs(vs[-1]) < 4e-05:
    #     print(i, j, vs[-1])

    # plt.plot(k_i_index,result,'-')
    # plt.plot(k_i_index,result,'r*')
    # plt.xlabel('k_i values')
    # plt.ylabel('v[-1] values')
    # plt.title('Daye hello')
    # plt.savefig("mygraph.png")
    
    # plt.plot(k_j_index,result,'-')
    # plt.plot(k_j_index,result,'r*')
    # plt.xlabel('k_j values')
    # plt.ylabel('v[-1] values')
    # plt.title('Daye hello')
    # plt.savefig("mygraph.png")
