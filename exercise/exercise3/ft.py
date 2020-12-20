import matplotlib.pyplot as plt
import cmath
import numpy as np

# 離散フーリエ変換
def dft(f):
    Y = [] # 結果
    n = len(f) # データ長
    for t in range(n):
        y = 0j
        for x in range(n):
            y += f[x] * cmath.e**(-2 * cmath.pi * 1j * x * t  / cmath.sqrt(n))
        Y.append(y) 
    return Y

# 離散フーリエ逆変換
def idft(f):
    Y = [] # 結果
    n = len(f) # データ長
    for t in range(n):
        y = 0j
        for x in range(n):
            y += f[x] * cmath.e**(2 * cmath.pi * 1j * x * t  / cmath.sqrt(n))
        Y.append(y) 
    return Y

# 高速フーリエ変換
def fft(f):
    n = len(f) # データ長

    # 再起動作の一番最後
    if n==1:
        return f[0]
    
    f_even = f[0:n:2]
    f_odd = f[1:n:2]
    
    # 再帰的に呼び出す
    Y_even = fft(f_even)
    Y_odd = fft(f_odd)
    
    # DFTと同じようにWを求める
    W = []
    for t in range(n//2):
        W.append(cmath.exp(-2 * cmath.pi * 1j * t) / cmath.sqrt(n))
    W = np.array(W)
    
    Y = np.zeros(n, dtype="complex")
    Y[0:n//2] = Y_even + W*Y_odd
    Y[n//2:n] = Y_even - W*Y_odd
    
    return Y

if __name__ == '__main__':
    # データ長は2の7乗
    N = 128
    x = np.arange(N)
    # 周期
    t1 = 10
    t2 = 20
    t3 = 30
    # 3つの周期の重ね合わせの合成関数
    y = np.sin(t1 * 2 * np.pi * (x/N)) + np.sin(t2 * 2 * np.pi * (x/N)) + np.sin(t3 * 2 * np.pi * (x/N))

    plt.plot(fft(y))
    plt.show()