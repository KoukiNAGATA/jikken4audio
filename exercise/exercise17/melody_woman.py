# ライブラリの読み込み
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ(2^12=4096)
size_frame = 2 ** 12

# フレームサイズに合わせて窓を作成
window = np.hamming(size_frame)

# シフトサイズ
size_shift = SR / 1000	# 0.001 秒 (10 msec)

# 音高の幅
low_pitch = 60
high_pitch = 71

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
	return int (round (12.0 * (np.log(frequency / 440.0) / np.log (2.0)))) + 69

# ノートナンバーから対応する全高調波成分の振幅の総和を計算
def calc_melody_likelihood(spectrum, notenum):
	fundamental_frequency = nn2hz(notenum)	#ノートナンバーに対応する周波数
	melody_likelihood = 0

	# 基本周波数
	s_idx = int(len(spectrum) * (fundamental_frequency / (SR/2)))
	melody_likelihood += spectrum[s_idx]
	# 2倍音
	s_idx = int(len(spectrum) * (fundamental_frequency * 2 / (SR/2)))
	melody_likelihood += spectrum[s_idx] * 0.75
	# 3倍音
	s_idx = int(len(spectrum) * (fundamental_frequency * 3 / (SR/2)))
	melody_likelihood += spectrum[s_idx] * 0.50

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
	x, _ = librosa.load("exercise/exercise17/woman.wav", sr=SR)

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

		# 対数振幅スペクトログラムから、ノートナンバーに対応する周波数の振幅の総和を取得
		frame_likelihood = 0
		frame_index = 0
		for i in range(low_pitch, high_pitch):
			melody_likelihood = calc_melody_likelihood(fft_abs_spec, i)
			if melody_likelihood > frame_likelihood :
				frame_index = i
				frame_likelihood = melody_likelihood

		melody.append(nn2hz(frame_index))

	#
	# 画像に表示・保存
	#

	# 画像として保存するための設定
	fig = plt.figure()
	# 余白を合わせる
	mpl.rcParams['axes.xmargin'] = 0
	mpl.rcParams['axes.ymargin'] = 0

	# まずは波形を描画
	ax1 = fig.add_subplot(211)	# 2行1列の1番目
	ax1.plot(x)
	ax1.set_xlabel('time [s/16000]')

    # 続いて下にスペクトログラムを描画
	ax2 = fig.add_subplot(212)	# 2行1列の2番目
	ax2.set_xlabel('frames')
	ax2.set_ylabel('frequency [Hz]')
	ax2.set_ylim([0, 500])
	ax2.imshow(
		np.flipud(np.array(spectrogram).T),
		extent=[0, len(melody), 0, SR/2],
		aspect='auto',
		interpolation='nearest'
	)

	# 最後に音高を描画
	ax2.plot(np.array(melody), color = "r")

	plt.show()

	# 保存
	fig.savefig('exercise/exercise17/woman.png')