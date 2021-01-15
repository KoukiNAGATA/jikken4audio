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

#####################################################################

if __name__ == "__main__":
	#
	# 入力テストデータの認識
	#

	# 母音の推定インデックスの格納場所
	predicted = []

	# スペクトログラムを保存するlist
	spectrogram = []

	# ゼロ交差数の格納場所
	zero_count = []

	# 音声ファイルの読み込み
	y, _ = librosa.load("exercise/exercise14/thomas.wav", sr=SR)