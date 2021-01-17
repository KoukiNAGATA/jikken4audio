# ライブラリの読み込み
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ(2^11=2048)
size_frame = 2 ** 11

# フレームサイズに合わせてブラックマン窓を作成
window = np.blackman(size_frame)

# シフトサイズ
size_shift = SR / 1000	# 0.001 秒 (10 msec)

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
	return int (round (12.0 * (np.log(frequency / 440.0) / np.log (2.0)))) + 69

# ノートナンバーから対応する周波数の振幅の総和を計算
def calc_melody_likelihood(spectrum, notenum, frequencies):
	# スペクトルの周波数ビン毎にクロマベクトルの対応する要素に振幅スペクトルを足しこむ
	melody_likelihood = 0
	amplitude_ratio = 1.0	# 倍音に減衰をかける
	fundamental_frequency = int(nn2hz(notenum))
	for s, f in zip (spectrum, frequencies) :
		if f % fundamental_frequency == 0 :
			melody_likelihood += np.abs(s) * amplitude_ratio
			amplitude_ratio *= 0.5
	return melody_likelihood

#####################################################################

if __name__ == "__main__":
	#
	# 入力テストデータの認識
	#

	# スペクトログラムを保存するlist
	spectrogram = []

	# メロディを保存するlist
	melody = []

	# 音声ファイルの読み込み
	x, _ = librosa.load("exercise/exercise17/shs-test-woman.wav", sr=SR)

	# 
	# 短時間フレームにおける処理
	# 

	for i in np.arange(0, len(x)-size_frame, size_shift):
		# 該当フレームのデータを取得
		idx = int(i)	# arangeのインデクスはfloatなのでintに変換
		x_frame = x[idx : idx+size_frame]

		# 窓掛けしたデータをFFT
		fft_spec = np.fft.rfft(x_frame * window)

		# 絶対値を取る
		fft_abs_spec = np.abs(fft_spec)

		# 複素スペクトログラムを対数振幅スペクトログラムに
		fft_log_abs_spec = np.log(fft_abs_spec)

		# 計算した対数振幅スペクトログラムを配列に保存
		spectrogram.append(fft_log_abs_spec)

		# 周波数ビンを取得
		frequencies = np.linspace(SR/len(fft_abs_spec), SR, len(fft_abs_spec))

		# 対数振幅スペクトログラムから、ノートナンバーに対応する周波数の振幅の総和を取得
		frame_likelihood = 0
		frame_index = 0
		for i in range(50, 70):
			melody_likelihood = calc_melody_likelihood(fft_abs_spec, i, frequencies)
			if melody_likelihood > frame_likelihood :
				frame_index = i
				frame_likelihood = melody_likelihood

		melody.append(frame_index)

	#
	# 画像に表示・保存
	#

	# 画像として保存するための設定
	fig = plt.figure()
	# 余白を合わせる
	mpl.rcParams['axes.xmargin'] = 0
	mpl.rcParams['axes.ymargin'] = 0

	# まずは波形を描画
	ax1 = fig.add_subplot(311)	# 3行1列の1番目
	ax1.plot(x)

    # 続いて下にスペクトログラムを描画
	ax2 = fig.add_subplot(312)	# 3行1列の2番目
	ax2.set_ylabel('frequency [Hz]')
	ax2.set_ylim([0, 500])
	ax2.imshow(
		np.flipud(np.array(spectrogram).T),
		extent=[0, len(x), 0, SR/2],
		aspect='auto',
		interpolation='nearest'
	)

	# 最後に右側のy軸を追加して，音高を描画
	ax3 = ax2.twinx()
	ax3.set_xlabel('time [s/16000]')
	ax3.set_ylabel('melody')
	ax3.plot(melody)

	plt.show()

	# 保存
	fig.savefig('exercise/exercise17/shs-test-woman.png')