# ライブラリの読み込み
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import librosa
import tkinter

# MatplotlibをTkinterで使用するために必要
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# サンプリングレート
SR = 16000

# フレームサイズ(2^12=4096)
size_frame = 2 ** 12

# フレームサイズに合わせて窓を作成
window = np.blackman(size_frame)

# シフトサイズ
size_shift = SR / 1000	# 0.001 秒 (10 msec)

# スペクトルを受け取り，ケプストラムの実数部分を返す関数
def calc_ceps(amplitude_spectrum):
	# 窓を作成
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

# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
	zc = 0

	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1

	return zc * SR / size_frame 	# 単位時間あたりに変換

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	if index == 0:
		return a[0] > a[1]
	elif index == len(a)-1:
		return a[index-1] < a[index]
	else:
		return a[index] > a[index-1] and a[index] > a[index+1]

#####################################################################

if __name__ == "__main__":
	# スペクトログラムを保存するlist
	spectrogram = []

	# 基本周波数を保存するlist
	frequency = []

	# 母音の推定インデックスの格納場所
	predicted = []

	# ゼロ交差数を保存するlist
	zero_count = []

	# 音声ファイルの読み込み
	y, _ = librosa.load("task1/waves/aiueo.wav", sr=SR)

    # ファイルサイズ（秒)
	duration = len(y) / SR

	# パラメータの読み込み
	mu_list = np.load('task1/parameter/mu.npz')
	mu_a = mu_list['arr_0']
	mu_i = mu_list['arr_1']
	mu_u = mu_list['arr_2']
	mu_e = mu_list['arr_3']
	mu_o = mu_list['arr_4']

	sigma_list = np.load('task1/parameter/sigma.npz')
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

		#
		# ゼロ交差数をプロット
		#

		zero_count.append(zero_cross(y_frame))

		#
		# 基本周波数をプロット
		#

		# 自己相関が格納された，長さが len(y_frame)*2-1 の対称な配列を得る
		autocorr = np.correlate(y_frame, y_frame, 'full')

		# 不要な前半を捨てる
		autocorr = autocorr[len(autocorr)//2 : ]

		# ピークのインデックスを抽出する
		peakindices = [i for i in range (len(autocorr)) if is_peak(autocorr, i)]

		# インデックス0 がピークに含まれていれば捨てる
		peakindices = [i for i in peakindices if i != 0]

		# 自己相関が最大となるインデックスを得る(単位:インデックス)
		max_peak_index = max(peakindices, key=lambda index: autocorr[index])

		# インデックスに対応する周波数を得る(単位:1/秒)
		frequency_frame = 1/(max_peak_index/SR)

		# リストに追加
		frequency.append(frequency_frame)

		#
		# スペクトログラムをプロット
		#

		# 窓掛けしたデータをFFT
		fft_spec = np.fft.rfft(y_frame * window)

		# 複素スペクトログラムを対数振幅スペクトログラムに
		fft_log_abs_spec = np.log(np.abs(fft_spec))

		# 計算した対数振幅スペクトログラムを配列に保存
		spectrogram.append(fft_log_abs_spec)

		#
		# 母音推定結果をプロット
		#

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

	# 閾値を設定。ゼロ交差数が範囲外の場合基本周波数を0とする。左右1フレームも0にした。
	for i in range(len(zero_count)):
		if zero_count[i] < 200 or zero_count[i] > 1000:
			if i == 0:
				frequency[i+1] = 0
			elif i == len(zero_count) - 1:
				frequency[i-1] = 0
			else :
				frequency[i+1] = 0
				frequency[i-1] = 0
			frequency[i] = 0

	# 1から始まるようにインデックス修正(あ=1, い=2, う=3, え=4, お=5)
	predicted = np.array(predicted) + 1


	# パラメータの保存
	np.savez('task1/parameter/lists.npz', spectrogram, frequency, predicted)

    # 画像として保存するための設定
	fig = plt.figure()

	# 余白を合わせる
	mpl.rcParams['axes.xmargin'] = 0
	mpl.rcParams['axes.ymargin'] = 0

	# まずはスペクトログラムを描画
	ax1 = fig.add_subplot(211)
	ax1.set_ylabel('frequency [Hz]')
	ax1.set_ylim([0, 500])
	ax1.imshow(
        np.flipud(np.array(spectrogram).T),
        extent=[0, len(predicted), 0, SR/2],
        aspect='auto',
        interpolation='nearest'
    )

    # 続いて基本周波数を重ねて描画
	ax1.plot(frequency, c = "r")

	# 続いて下に母音推定結果を描画
	ax2 = fig.add_subplot(212)
	ax2.set_xlabel('frames')
	ax2.set_ylabel('predicted')
	ax2.set_ylim([0, 6])
	ax2.plot(predicted)

	plt.show()
	fig.savefig('task1/images/plot-info.png')