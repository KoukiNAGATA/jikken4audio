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

# 周波数を保存するlist
frequency = []

# スペクトログラムを保存するlist
spectrogram = []

# ゼロ交差数を保存するlist
zero_count = []

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

# 音声ファイルの読み込み
x, _ = librosa.load('exercise/exercise14/voice.wav', sr=SR)

# 
# 短時間フレームにおける処理
# 

for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]

	#
	# ゼロ交差数をプロット
	#

	zero_count.append(zero_cross(x_frame))

	#
	# 基本周波数をプロット
	#

	# 自己相関が格納された，長さが len(x_frame)*2-1 の対称な配列を得る
	autocorr = np.correlate(x_frame, x_frame, 'full')

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
	fft_spec = np.fft.rfft(x_frame * window)

	# 複素スペクトログラムを対数振幅スペクトログラムに
	fft_log_abs_spec = np.log(np.abs(fft_spec))

	# 計算した対数振幅スペクトログラムを配列に保存
	spectrogram.append(fft_log_abs_spec)

# 閾値を設定。ゼロ交差数が0の場合基本周波数を0とする。左右1フレームも0にした。
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

#
# 画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

plt.xlabel('frames')							# x軸のラベルを設定
plt.ylabel('fundamental frequency [Hz]')		# y軸のラベルを設定
# スペクトログラムを描画
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転地して上下反転
	extent=[0, len(frequency), 0, SR/2],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)

plt.plot(frequency)
plt.ylim([0, 1000])    # 縦軸を拡大する。
plt.show()

# 保存
fig.savefig('exercise/exercise14/voice.png')
