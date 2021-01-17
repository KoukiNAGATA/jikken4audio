# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 4096

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

#
# ケプストラム集合の対数尤度を返す関数
# 	x:ケプストラム係数 (D)
# 	mu:ある母音のmu (D)
# 	sigma:ある母音のsigma (D)
#
def calc_likelihood(x, mu, sigma):
	eps = 1e-8
	results = 0
	for i in range(x.shape[0]):
		results += -(np.log(np.sqrt(sigma[i]) + eps) + (x[i] - mu[i])**2/(2*sigma[i] + eps))
	return results

#####################################################################

if __name__ == "__main__":
	#
	# 入力テストデータの認識
	#

	# 母音の推定インデックスの格納場所
	predicted = []

	# スペクトログラムを保存するlist
	spectrogram = []

	# 音声ファイルの読み込み
	y, _ = librosa.load("waves/continuous/aiueo.wav", sr=SR)

	# パラメータの読み込み
	mu_list = np.load('exercise/exercise15/mu.npz')
	mu_a = mu_list['arr_0']
	mu_i = mu_list['arr_1']
	mu_u = mu_list['arr_2']
	mu_e = mu_list['arr_3']
	mu_o = mu_list['arr_4']

	sigma_list = np.load('exercise/exercise15/sigma.npz')
	sigma_a = sigma_list['arr_0']
	sigma_i = sigma_list['arr_1']
	sigma_u = sigma_list['arr_2']
	sigma_e = sigma_list['arr_3']
	sigma_o = sigma_list['arr_4']

	# 
	# 短時間フレームにおける処理
	# 

	for i in np.arange(0, len(y)-size_frame, size_shift):
		# 該当フレームのデータを取得
		idx = int(i)	# arangeのインデクスはfloatなのでintに変換
		y_frame = y[idx : idx+size_frame]

		# 窓掛けしたデータをFFT
		fft_spec = np.fft.rfft(y_frame * window)

		# 複素スペクトログラムを対数振幅スペクトログラムに
		fft_log_abs_spec = np.log(np.abs(fft_spec))

		# 計算した対数振幅スペクトログラムを配列に保存
		spectrogram.append(fft_log_abs_spec)

		# スペクトルを受け取り，ケプストラムを得る
		frame_cepstrum = calc_ceps(fft_log_abs_spec)

		# 対数尤度の計算
		likelihood_a = calc_likelihood(frame_cepstrum, mu_a, sigma_a)
		likelihood_i = calc_likelihood(frame_cepstrum, mu_i, sigma_i)
		likelihood_u = calc_likelihood(frame_cepstrum, mu_u, sigma_u)
		likelihood_e = calc_likelihood(frame_cepstrum, mu_e, sigma_e)
		likelihood_o = calc_likelihood(frame_cepstrum, mu_o, sigma_o)
		likelihood_list = [likelihood_a, likelihood_i, likelihood_u, likelihood_e, likelihood_o]
		# 最大値のインデックスを取得(あ=0, い=1, う=2, え=3, お=4)
		predicted.append(likelihood_list.index(max(likelihood_list)))

	# 1から始まるようにインデックス修正(あ=1, い=2, う=3, え=4, お=5)
	predicted = np.array(predicted) + 1

	#
	# 画像に表示・保存
	#

	# 画像として保存するための設定
	fig = plt.figure()

	# まずはスペクトログラムを描画
	ax1 = fig.add_subplot(111)
	ax1.set_xlabel('frames')
	ax1.set_ylabel('frequency [Hz]')
	ax1.imshow(
		np.flipud(np.array(spectrogram).T),
		extent=[0, len(predicted), 0, SR/2],
		aspect='auto',
		interpolation='nearest'
	)

	# 続いて右側のy軸を追加して，母音を重ねて描画
	ax2 = ax1.twinx()
	ax2.set_ylabel('predicted')
	ax2.plot(predicted)

	plt.show()

	# 保存
	fig.savefig('exercise/exercise15/predicted.png')