# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 512

# フレームサイズに合わせてブラックマン窓を作成
window = np.blackman(size_frame)

# シフトサイズ
size_shift = 16000 / 1000	# 0.001 秒 (10 msec)

# スペクトルを受け取り，ケプストラムの実数部分を返す関数
def calc_ceps(amplitude_spectrum):
	# ブラックマン窓を作成
	window = np.blackman(len(amplitude_spectrum))
	# 窓掛けしたデータをFFT
	cepstrum = np.fft.rfft(amplitude_spectrum * window)
	# 13次までのケプストラム係数を抽出
	coefficients = cepstrum[:13]
	# 0埋め
	cepstrum = np.append(coefficients, [0] * (len(cepstrum) - len(coefficients)))
	# 取り出した成分を逆フーリエ変換し、実数部分を取り出す
	spectral_envelope = np.abs(np.fft.irfft(cepstrum))
	return spectral_envelope

# 短時間フレームにおけるケプストラム計算
def get_ceps(x, ceps_x):
	for i in np.arange(0, len(x)-size_frame, size_shift):
		# 該当フレームのデータを取得
		idx = int(i)	# arangeのインデクスはfloatなのでintに変換
		x_frame = x[idx : idx+size_frame]

		# 窓掛けしたデータをFFT
		fft_spec = np.fft.rfft(x_frame * window)

		# 複素スペクトログラムを対数振幅スペクトログラムに
		fft_log_abs_spec = np.log(np.abs(fft_spec))

		# スペクトルを受け取り，ケプストラムを得る
		frame_cepstrum = calc_ceps(fft_log_abs_spec)

		# ケプストラムを配列に保存
		ceps_x.append(frame_cepstrum)
	return ceps_x

#####################################################################

if __name__ == "__main__":
	# 
	# 学習させるデータの多次元正規分布を学習
	# 

	# 音声ファイルの読み込み
	x_a, _ = librosa.load('waves/discrete/a.wav', sr=SR)
	x_i, _ = librosa.load('waves/discrete/i.wav', sr=SR)
	x_u, _ = librosa.load('waves/discrete/u.wav', sr=SR)
	x_e, _ = librosa.load('waves/discrete/e.wav', sr=SR)
	x_o, _ = librosa.load('waves/discrete/o.wav', sr=SR)

	# 「あ」「い」「う」「え」「お」のケプストラム計算
	ceps_a = []
	ceps_i = []
	ceps_u = []
	ceps_e = []
	ceps_o = []

	ceps_a = get_ceps(x_a, ceps_a)
	ceps_i = get_ceps(x_i, ceps_i)
	ceps_u = get_ceps(x_u, ceps_u)
	ceps_e = get_ceps(x_e, ceps_e)
	ceps_o = get_ceps(x_o, ceps_o)

	# 平均
	mu_a = np.mean(np.array(ceps_a), axis=0)
	mu_i = np.mean(np.array(ceps_i), axis=0)
	mu_u = np.mean(np.array(ceps_u), axis=0)
	mu_e = np.mean(np.array(ceps_e), axis=0)
	mu_o = np.mean(np.array(ceps_o), axis=0)

	# 平均二乗誤差
	sigma_a = np.mean((np.array(ceps_a) - mu_a[None,:])**2, axis=0)
	sigma_i = np.mean((np.array(ceps_i) - mu_i[None,:])**2, axis=0)
	sigma_u = np.mean((np.array(ceps_u) - mu_u[None,:])**2, axis=0)
	sigma_e = np.mean((np.array(ceps_e) - mu_e[None,:])**2, axis=0)
	sigma_o = np.mean((np.array(ceps_o) - mu_o[None,:])**2, axis=0)

	# パラメータの保存
	np.savez('exercise/exercise15/mu.npz', mu_a, mu_i, mu_u, mu_e, mu_o)
	np.savez('exercise/exercise15/sigma.npz', sigma_a, sigma_i, sigma_u, sigma_e, sigma_o)
