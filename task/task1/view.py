# 
# gui.pyが実行時間がかかるのでGUI部分のみ確認用
#

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

if __name__ == "__main__":
	# スペクトログラムを保存するlist
	spectrogram = []

	# 基本周波数を保存するlist
	frequency = []

	# 母音の推定インデックスの格納場所
	predicted = []

	# ゼロ交差数を保存するlist
	zero_count = []

	# パラメータの読み込み
	lists = np.load('task1/parameter/lists.npz')
	spectrogram = lists['arr_0']
	frequency = lists['arr_1']
	predicted = lists['arr_2']

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