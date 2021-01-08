# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('waves/continuous/aiueo.wav', sr=SR)

# フーリエ変換
fft_spec = np.fft.fft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに
fft_log_abs_spec = np.log(np.abs(fft_spec))

# スペクトルを画像に表示・保存

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('Amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
plt.plot(fft_log_abs_spec)			# 描画

# 表示
plt.show()

# 画像ファイルに保存
fig.savefig('images/exercise2/whole.png')