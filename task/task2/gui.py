# ライブラリの読み込み
import pyaudio
import numpy as np
import threading
import time

# matplotlib関連
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 4096

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE / 20)	# 今回は0.05秒

# スペクトルをカラー表示する際に色の範囲を正規化するために
# スペクトルの最小値と最大値を指定
# スペクトルの値がこの範囲を超えると，同じ色になってしまう
SPECTRUM_MIN = -5
SPECTRUM_MAX = 10

# log10を計算する際に，引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 150

# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False

# 音高の幅
LOW_PITCH = 55
HIGH_PITCH = 75

# 周波数の範囲
LOW_FREQUENCY = 0
HIGH_FREQUENCY = 1500

#
# (1) GUI / グラフ描画の処理
#

# ここでは matplotlib animation を用いて描画する
# 毎回 figure や ax を初期化すると処理時間がかかるため
# データを更新したら，それに従って必要な部分のみ再描画することでリアルタイム処理を実現する

# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う
def animate(frame_index):

	ax1_sub.set_array(spectrogram_data)
	ax2_sub.set_data(time_x_data, sing_data)	# マイク入力の音高
	ax3_sub.set_data(time_x_data, melody_data)	# 楽曲の音高
	
	return ax1_sub, ax2_sub, ax3_sub

# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("粗雑採点DX(採点はしません)")

# スペクトログラムを描画
fig, ax1 = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=root)

# 横軸の値のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
# 縦軸の値のデータ
freq_y_data = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)

# とりあえず初期値（ゼロ）のスペクトログラムと音高のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
sing_data = np.zeros(len(time_x_data))
melody_data = np.zeros(len(time_x_data))

# スペクトログラムを描画する際に横軸と縦軸のデータを行列にしておく必要がある
# これは下記の matplotlib の pcolormesh の仕様のため
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
	for idx_t, t_v in enumerate(time_x_data):
		X[idx_f, idx_t] = t_v
		Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub = ax1.pcolormesh(
	X,
	Y,
	spectrogram_data,
	shading='nearest',	# 描画スタイル
	cmap='BuPu',			# カラーマップ
	alpha=0.7, 	# 透明度
	norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)	# 値の最小値と最大値を指定して，それに色を合わせる
)

# 音高を表示するために反転した軸を作成
ax2 = ax1.twinx()
ax3 = ax1.twinx()

# 音高をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax2_sub, = ax2.plot(time_x_data, sing_data, c='y', linewidth = 5.0)
ax3_sub, = ax3.plot(time_x_data, melody_data, c='#D3D3D3', linewidth = 5.0)

# ラベルの設定
ax1.set_xlabel('sec')				# x軸のラベルを設定
ax1.set_ylabel('frequency [Hz]')	# y軸のラベルを設定

# y軸の範囲の設定
ax1.set_ylim([LOW_FREQUENCY, HIGH_FREQUENCY])
ax2.set_ylim([LOW_FREQUENCY, HIGH_FREQUENCY])
ax3.set_ylim([LOW_FREQUENCY, HIGH_FREQUENCY])

# maplotlib animationを設定
ani = animation.FuncAnimation(
	fig,
	animate,		# 再描画のために呼び出される関数
	interval=100,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	blit=True		# blitting処理を行うため描画処理が速くなる
)

# matplotlib を GUI(Tkinter) に追加する
toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master=root, textvariable=text, font=("", 30))
label.pack()

# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
	root.quit()
	root.destroy()

# 終了ボタンを作成
button = tkinter.Button(master=root, text="終了", command=_quit, font=("", 30))
button.pack()


#
# (2) マイク入力のための処理
#

x_stacked_data = np.array([])

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
	s_idx = int(len(spectrum) * (fundamental_frequency / (SAMPLING_RATE/2)))
	melody_likelihood += spectrum[s_idx]
	# 2倍音
	s_idx = int(len(spectrum) * (fundamental_frequency * 2 / (SAMPLING_RATE/2)))
	melody_likelihood += spectrum[s_idx] * 0.75
	# 3倍音
	s_idx = int(len(spectrum) * (fundamental_frequency * 3 / (SAMPLING_RATE/2)))
	melody_likelihood += spectrum[s_idx] * 0.50

	return melody_likelihood

# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):

	zc = 0

	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1

	return zc * SAMPLING_RATE / FRAME_SIZE 	# 単位時間あたりに変換

# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):
	
	# この関数は別スレッドで実行するため
	# メインスレッドで定義した以下の numpy array を利用できるように global 宣言する
	# これらにはフレーム毎のスペクトルと音量のデータが格納される
	global x_stacked_data, sing_data

	# 現在のフレームの歌声データをnumpy arrayに変換
	x_current_frame = np.frombuffer(in_data, dtype=np.float32)

	# 現在のフレームとこれまでに入力されたフレームを連結
	x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

	# フレームサイズ分のデータがあれば処理を行う
	if len(x_stacked_data) >= FRAME_SIZE:
		
		# フレームサイズからはみ出した過去のデータは捨てる
		x_stacked_data = x_stacked_data[len(x_stacked_data)-FRAME_SIZE:]

		# スペクトルを計算
		fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
		fft_abs_spec = np.abs(fft_spec)

		# 対数振幅スペクトログラムから、ノートナンバーに対応する周波数の振幅の総和を取得
		frame_likelihood = 0
		frame_index = 0
		for i in range(LOW_PITCH, HIGH_PITCH):
			melody_likelihood = calc_melody_likelihood(fft_abs_spec, i)
			if melody_likelihood > frame_likelihood :
				frame_index = i
				frame_likelihood = melody_likelihood

		sing_data = np.roll(sing_data, -1)
		# 音量が閾値未満ならNanを配列に追加し、グラフ内に表示させない
		vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
		if vol < -100 :
			sing_data[-1] = np.nan
		else :
			sing_data[-1] = nn2hz(frame_index)
	
	# 戻り値は pyaudio の仕様に従うこと
	return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
# 【注意】シフトサイズごとに指定された関数が呼び出される
p = pyaudio.PyAudio()
stream = p.open(
	format = pyaudio.paFloat32,
	channels = 1,
	rate = SAMPLING_RATE,
	input = True,						# ここをTrueにするとマイクからの入力になる 
	frames_per_buffer = SHIFT_SIZE,		# シフトサイズ
	stream_callback = input_callback	# ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) mp3ファイル音楽を再生する処理
#

audio_stacked_data = np.array([])

# mp3ファイル名
# ここは各自の音源ファイルに合わせて変更すること
filename = './mp3/hotaru_no_hikari.mp3'

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_mp3(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
	format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
	channels = audio_data.channels,								# チャネル数
	rate = audio_data.frame_rate,								# サンプリングレート
	output = True												# 出力モードに設定
)

# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def play_music():

	# この関数は別スレッドで実行するため
	# メインスレッドで定義した以下の変数を利用できるように global 宣言する
	global is_gui_running, audio_data, now_playing_sec
	global spectrogram_data, audio_stacked_data, melody_data

	# pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
	# 第二引数には何ミリ秒毎に読み込むかを指定

	size_frame_music = 100	# 100ミリ秒毎に読み込む

	idx = 0

	# make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む

	# フレーム毎に計算し、楽曲のスペクトログラム配列に加える
	# 再生位置も計算する
	for chunk in make_chunks(audio_data, size_frame_music):
		
		# GUIが終了してれば，この関数の処理も終了する
		if is_gui_running == False:
			break

		# 現在のフレームの楽曲データをnumpy arrayに変換
		audio_current_frame = chunk.get_array_of_samples()

		# 片側だけの音を抽出
		audio_current_frame = audio_current_frame[::chunk.channels]

		# 現在のフレームとこれまでに入力されたフレームを連結
		audio_stacked_data = np.concatenate([audio_stacked_data, audio_current_frame])
		
		# フレームサイズ分のデータがあれば処理を行う
		if len(audio_stacked_data) >= FRAME_SIZE:

			# フレームサイズからはみ出した分は捨てる
			audio_stacked_data = audio_stacked_data[len(audio_stacked_data)-FRAME_SIZE:]

			# スペクトルを計算
			fft_spec = np.fft.rfft(audio_stacked_data * hamming_window)
			fft_abs_spec = np.abs(fft_spec)
			fft_log_abs_spec = np.log10(fft_abs_spec + EPSILON)[:-1]

			# ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
			# 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
			spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
			spectrogram_data[:, -1] = fft_log_abs_spec

			# 対数振幅スペクトログラムから、ノートナンバーに対応する周波数の振幅の総和を取得
			frame_likelihood = 0
			frame_index = 0
			for i in range(LOW_PITCH, HIGH_PITCH):
				melody_likelihood = calc_melody_likelihood(fft_abs_spec, i)
				if melody_likelihood > frame_likelihood :
					frame_index = i
					frame_likelihood = melody_likelihood
			melody_data = np.roll(melody_data, -1)
			melody_data[-1] = nn2hz(frame_index)

			melody_data = np.roll(melody_data, -1)
			# ゼロ交差数をカウント
			zero_count = zero_cross(audio_stacked_data * hamming_window)
			# 閾値を設定。ゼロ交差数が範囲外の場合基本周波数を0とする。
			if zero_count < 800 or zero_count > 3000:
				melody_data[-1] = np.nan
			else :
				melody_data[-1] = nn2hz(frame_index)

		# pyaudioの再生ストリームに切り出した音楽データを流し込む
		# 再生が完了するまで処理はここでブロックされる
		stream_play.write(chunk._data)

		# 現在の再生位置を計算（単位は秒）
		now_playing_sec = (idx * size_frame_music) / 1000.
		
		idx += 1

# 再生時間の表示を随時更新する関数
def update_gui_text():

	global is_gui_running, now_playing_sec, text

	while True:
		
		# GUIが表示されていれば再生位置（秒）をテキストとしてGUI上に表示
		if is_gui_running:
			text.set('%.3f' % now_playing_sec)
		
		# 0.01秒ごとに更新
		time.sleep(0.01)

# 再生時間を表す
now_playing_sec = 0.0

# 音楽を再生するパートを関数化したので，それを別スレッドで（GUIのため）再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため
t_play_music.start()

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため
t_update_gui.start()

#
# (4) 全体の処理を実行
#

# GUIの開始フラグをTrueに
is_gui_running = True

# GUIを開始，GUIが表示されている間は処理はここでストップ
tkinter.mainloop()

# GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
is_gui_running = False

# 終了処理
stream_play.stop_stream()
stream_play.close()
p_play.terminate()