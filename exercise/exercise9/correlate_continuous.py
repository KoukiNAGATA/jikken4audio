# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# フレームサイズ
size_frame = 512			# 2のべき乗

# シフトサイズ
size_shift = 16000 / 1000	# 0.001 秒 (10 msec)

# 周波数を保存するlist
frequency = []

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	if index == 0:
		return a[0] > a[1]
	elif index == len(a)-1:
		return a[index-1] < a[index]
	else:
		return a[index] > a[index-1] and a[index] > a[index+1]

# 音声ファイルの読み込み
x, _ = librosa.load('waves/continuous/aiueo.wav', sr=SR)

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]

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
# 基本周波数を画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# 基本周波数を描画
plt.xlabel('frames')							# x軸のラベルを設定
plt.ylabel('fundamental frequency [Hz]')		# y軸のラベルを設定
plt.plot(frequency)			# 描画
plt.show()

# 基本周波数を保存
fig.savefig('images/exercise9/fundamental_frequency_aiueo_continuous.png')