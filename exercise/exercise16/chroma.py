# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math

# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 512

# フレームサイズに合わせてブラックマン窓を作成
window = np.blackman(size_frame)

# シフトサイズ
size_shift = 16000 / 1000	# 0.001 秒 (10 msec)

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69

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
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += math.abs(s)
	
	return cv

#####################################################################

if __name__ == "__main__":
	#
	# 入力テストデータの認識
	#

	# スペクトログラムを保存するlist
	spectrogram = []

	# ゼロ交差数の格納場所
	zero_count = []

	# 音声ファイルの読み込み
	x, _ = librosa.load("exercise/exercise16/thomas.wav", sr=SR)

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

		# 複素スペクトログラムを対数振幅スペクトログラムに
		fft_log_abs_spec = np.log(np.abs(fft_spec))

		# 計算した対数振幅スペクトログラムを配列に保存
		spectrogram.append(fft_log_abs_spec)


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

	# 続いて右側のy軸を追加して，音量を重ねて描画
	ax2 = ax1.twinx()
	ax2.set_ylabel('predicted')
	ax2.plot(predicted)

	plt.show()

	# 保存
	fig.savefig('exercise/exercise15/predicted.png')