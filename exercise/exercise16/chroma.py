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

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
	return int (round (12.0 * (np.log(frequency / 440.0) / np.log (2.0)))) + 69

#
# スペクトルと対応する周波数ビンの情報を受け取り，クロマベクトルを算出
#
# 【周波数ビンの情報について補足】
# 例えば，サンプリング周波数が16000の場合は，spectrumは8000Hzまでの情報を保持していることになるため，
# spectrumのサイズが512だとすると，
# frequencies = [1 * (8000/512), 2 * (8000/512), ..., 511 * (8000/512), 8000] とすればよい
# このような処理をnumpyで実現するばらば，
# frequencies = np.linspace(8000/len(spectrum), 8000, len(spectrum)) などどすればよい
#
def chroma_vector(spectrum, frequencies):

	# 0 = C, 1 = C#, 2 = D, ..., 11 = B
	# 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
	cv = np.zeros(12)
	
	# スペクトルの周波数ビン毎にクロマベクトルの対応する要素に振幅スペクトルを足しこむ
	for s, f in zip (spectrum, frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += np.abs(s)
	
	return cv

# コードのテンプレートベクトル
root_sound = 1.0
third_sound = 0.5
fifth_sound = 0.8
chord_dic = []
chord_dic.append([root_sound, 0,0,0, third_sound, 0,0, fifth_sound, 0,0,0,0])
chord_dic.append([0, root_sound, 0,0,0, third_sound, 0,0, fifth_sound, 0,0,0])
chord_dic.append([0,0, root_sound, 0,0,0, third_sound, 0,0, fifth_sound, 0,0])
chord_dic.append([0,0,0, root_sound, 0,0,0, third_sound, 0,0, fifth_sound, 0])
chord_dic.append([0,0,0,0, root_sound, 0,0,0, third_sound, 0,0, fifth_sound])
chord_dic.append([fifth_sound, 0,0,0,0, root_sound, 0,0,0, third_sound, 0,0])
chord_dic.append([0, fifth_sound, 0,0,0,0, root_sound, 0,0,0, third_sound, 0])
chord_dic.append([0,0, fifth_sound, 0,0,0,0, root_sound, 0,0,0, third_sound])
chord_dic.append([third_sound, 0,0, fifth_sound, 0,0,0,0, root_sound, 0,0,0])
chord_dic.append([0, third_sound, 0,0, fifth_sound, 0,0,0,0, root_sound, 0,0])
chord_dic.append([0,0, third_sound, 0,0, fifth_sound, 0,0,0,0, root_sound, 0])
chord_dic.append([0,0,0, third_sound, 0,0, fifth_sound, 0,0,0,0, root_sound])
chord_dic.append([root_sound, 0,0, third_sound, 0,0,0, fifth_sound, 0,0,0,0])
chord_dic.append([0, root_sound, 0,0, third_sound, 0,0,0, fifth_sound, 0,0,0])
chord_dic.append([0,0, root_sound, 0,0, third_sound, 0,0,0, fifth_sound, 0,0])
chord_dic.append([0,0,0, root_sound, 0,0, third_sound, 0,0,0, fifth_sound, 0])
chord_dic.append([0,0,0,0, root_sound, 0,0, third_sound, 0,0,0, fifth_sound])
chord_dic.append([fifth_sound, 0,0,0,0, root_sound, 0,0, third_sound, 0,0,0])
chord_dic.append([0, fifth_sound, 0,0,0,0, root_sound, 0,0, third_sound, 0,0])
chord_dic.append([0,0, fifth_sound, 0,0,0,0, root_sound, 0,0, third_sound, 0])
chord_dic.append([0,0,0, fifth_sound, 0,0,0,0, root_sound, 0,0, third_sound])
chord_dic.append([third_sound, 0,0,0, fifth_sound, 0,0,0,0, root_sound, 0,0])
chord_dic.append([0, third_sound, 0,0,0, fifth_sound, 0,0,0,0, root_sound, 0])
chord_dic.append([0,0, third_sound, 0,0,0, fifth_sound, 0,0,0,0, root_sound])

#####################################################################

if __name__ == "__main__":
	#
	# 入力テストデータの認識
	#

	# スペクトログラムを保存するlist
	spectrogram = []

	# クロマグラムを保存するlist
	chromagram = []

	# 和音を保存するlist
	chord = []

	# 音声ファイルの読み込み
	x, _ = librosa.load("exercise/exercise16/easy_chords.wav", sr=SR)

    # 楽音成分とパーカッシブ成分に分ける。楽音成分のみ用いる。
	x, _ = librosa.effects.hpss(x)

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

		# クロマベクトルを取得
		chroma_frame = chroma_vector(fft_abs_spec, frequencies)

		# 計算したクロマベクトルを配列に保存
		chromagram.append(chroma_frame)

		# クロマベクトルとテンプレートベクトルの内積を計算し、最大値を得る
		chord_likelihood = 0
		chord_frame = 0
		for i in range(24) :
			if np.dot(chord_dic[i], chroma_frame) > chord_likelihood :
				chord_frame = i
				chord_likelihood = np.dot(chord_dic[i], chroma_frame)
		chord.append(chord_frame)

	#
	# 画像に表示・保存
	#

	# 画像として保存するための設定
	fig = plt.figure()
	# 余白を合わせる
	mpl.rcParams['axes.xmargin'] = 0
	mpl.rcParams['axes.ymargin'] = 0

	# まずはスペクトログラムを描画
	ax1 = fig.add_subplot(311)	# 3行1列の1番目
	ax1.set_ylabel('frequency [Hz]')
	ax1.set_ylim([0, 4000])
	ax1.imshow(
		np.flipud(np.array(spectrogram).T),
		extent=[0, len(chord), 0, SR/2],
		aspect='auto',
		interpolation='nearest'
	)

	# 続いて下にクロマグラムを描画
	ax2 = fig.add_subplot(312)	# 3行1列の2番目
	ax2.set_ylabel('chromagram')
	ax2.imshow(
		np.flipud(np.array(chromagram).T),
		extent=[0, len(chord), 0, 11],
		aspect='auto',
		interpolation='nearest'
	)

	# 続いて下に和音を描画
	ax3 = fig.add_subplot(313)	# 3行1列の3番目
	ax3.set_xlabel('frames')
	ax3.set_ylabel('chord')
	ax3.plot(chord)

	plt.show()

	# 保存
	fig.savefig('exercise/exercise16/easy_chords.png')