# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 20000

# フレームサイズ
size_frame = 512			# 2のべき乗

# シフトサイズ
size_shift = 20000 / 1000	# 0.001 秒 (10 msec)

# フレーム単位の振幅を保存するlist
amplitude = []


# 振幅値のRMSを求める
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp)))

if __name__ == '__main__':
    # 音声ファイルの読み込み
    x, _ = librosa.load('waves/aiueo_discrete.wav', sr=SR)

    # size_shift分ずらしながらsize_frame分のデータを取得
    # np.arange関数はfor文で辿りたい数値のリストを返す
    # 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
    # (初期位置, 終了位置, 1ステップで進める間隔)
    for i in np.arange(0, len(x)-size_frame, size_shift):
        
        # 該当フレームのデータを取得
        idx = int(i)	# arangeのインデクスはfloatなのでintに変換
        x_frame = cal_rms(x[idx : idx+size_frame])
        amplitude.append(np.max(x_frame))
    
    # デシベルに変換
    volume = 20 * np.log10(amplitude)

    # 画像として保存するための設定
    fig = plt.figure()

    # スペクトログラムを描画
    plt.xlabel('time')		# x軸のラベルを設定
    plt.ylabel('volume [db]')				# y軸のラベルを設定
    plt.xlim([0, SR/2])					# x軸の範囲を設定
    plt.plot(volume)			# 描画

    # 表示
    plt.show()

    # 画像ファイルに保存
    fig.savefig('images/exercise6/volume_discrete.png')