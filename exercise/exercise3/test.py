# ライブラリの読み込み
import matplotlib.pyplot as plt
import cmath
import numpy as np
import librosa
import time

import ft

# サンプリングレート
SR = 16000

# データ長は2の7乗
N = 128
x = np.arange(N)
# 周期
t1 = 10
t2 = 20
t3 = 30
# 3つの周期の重ね合わせの合成関数
y = np.sin(t1 * 2 * np.pi * (x/N)) + np.sin(t2 * 2 * np.pi * (x/N)) + np.sin(t3 * 2 * np.pi * (x/N))

if __name__ == '__main__':
    # 離散フーリエ変換
    start = time.perf_counter()
    dft_spec = ft.dft(y)
    end = time.perf_counter()
    print(f"離散フーリエ変換: {end - start:.3f} s.")
    # 高速フーリエ変換
    start = time.perf_counter()
    fft_spec = ft.fft(y)
    end = time.perf_counter()
    print(f"高速フーリエ変換: {end - start:.3f} s.")