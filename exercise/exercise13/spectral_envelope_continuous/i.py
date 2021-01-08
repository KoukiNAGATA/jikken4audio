# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	# ブラックマン窓を作成
	window = np.blackman(len(amplitude_spectrum))
	# 窓掛けしたデータをFFT
	cepstrum = np.fft.rfft(amplitude_spectrum * window)
	# 13次までのケプストラム係数を抽出
	coefficients = cepstrum[:13]
	# 0埋め
	cepstrum = np.append(coefficients, [0] * (len(cepstrum) - len(coefficients)))
	# 取り出した成分を逆フーリエ変換する
	spectral_envelope = np.fft.irfft(cepstrum)
	return spectral_envelope

# 音声ファイルの読み込み
x, _ = librosa.load('waves/continuous/i.wav', sr=SR)

# フーリエ変換
fft_spec = np.fft.rfft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに(元の対数振幅スペクトル)
fft_log_abs_spec = np.log(np.abs(fft_spec))

# スペクトルを受け取り，ケプストラムを得る
x_cepstrum = cepstrum(fft_log_abs_spec)

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

# 保存
fig.savefig('images/exercise13/spectral_envelope_continuous/i.png')