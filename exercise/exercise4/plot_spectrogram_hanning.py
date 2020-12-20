# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('waves/aiueo_continuous.wav', sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 512			# 2のべき乗

# フレームサイズに合わせてハニング窓を作成
window = np.hanning(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# スペクトログラムを保存するlist
spectrogram = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]

	# 窓掛けしたデータをFFT
	# np.fft.rfftを使用するとFFTの前半部分のみが得られる
	fft_spec = np.fft.rfft(x_frame * window)

	# 複素スペクトログラムを対数振幅スペクトログラムに
	fft_log_abs_spec = np.log(np.abs(fft_spec))

	# 計算した対数振幅スペクトログラムを配列に保存
	spectrogram.append(fft_log_abs_spec)


#
# スペクトログラムを画像に表示・保存
#

# 画像として保存するための設定
fig = plt.figure()

# スペクトログラムを描画
plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
plt.imshow(
	np.flipud(np.array(spectrogram).T),		# 画像とみなすために，データを転地して上下反転
	extent=[0, len(x), 0, SR/2],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
	aspect='auto',
	interpolation='nearest'
)
plt.show()

# ハニング窓を保存
fig.savefig('images/exercise4/plot-spectogram-hanning.png')

