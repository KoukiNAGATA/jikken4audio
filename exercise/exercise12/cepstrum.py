# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 512			# 2のべき乗

# フレームサイズに合わせてブラックマン窓を作成
window = np.blackman(size_frame)

# シフトサイズ
size_shift = 16000 / 1000	# 0.001 秒 (10 msec)

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum

# 音声ファイルの読み込み
x, _ = librosa.load('waves/aiueo_continuous.wav', sr=SR)

# フーリエ変換
fft_spec = np.fft.fft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに(元の対数振幅スペクトル)
fft_log_abs_spec = np.log(np.abs(fft_spec))

# スペクトルを受け取り，ケプストラムを得る
x_cepstrum = cepstrum(x)

# スペクトルを画像に表示

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('Amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
plt.plot(fft_log_abs_spec)			# 描画
plt.plot(x_cepstrum)

# 表示
plt.show()