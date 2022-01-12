# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten,
                                    Input, Reshape)
from tensorflow.keras.models import Model


# scikit-learn の顔写真データをロード
digits = sklearn.datasets.fetch_olivetti_faces()
data = digits['data']
data = data / data.max()
# 入力画像の表示（64個）

for j in range(4):
    plt.figure( figsize=(8, 8))
    for k in range( 100 ):
        plt.subplot(10, 10,k+1) # 8x8 の subplot window の左上からk+1番目
        plt.xticks([])
        plt.yticks([])
        plt.imshow( data[100 * j + k].reshape(64, 64), cmap='gray', interpolation='None' )
    plt.savefig("Input_images{}.png".format(j + 1))
input()
label = np.array([k // 10 for k in range(100)])
latent = []
color = ['black','red','blue','green','magenta','cyan','gold','saddlebrown','pink','salmon' ]

for s in range(2, 11):
    # scikit-learn の顔写真データをロード
    digits = sklearn.datasets.fetch_olivetti_faces()
    data = digits['data'][300:]
    data = data / data.max()
    # s = 10 #潜在変数

    # Encoder
    encoder_input = Input(shape=data.shape[1], name='encoder_input')
    x1 = Reshape( (64, 64, 1) )( encoder_input )
    x2 = Conv2D(  64, ( 3, 3 ), padding='same', strides=(2,2), use_bias=True, activation='relu' )( x1 )
    x3 = Conv2D( 128, ( 3, 3 ), padding='same', strides=(2,2), use_bias=True, activation='relu' )( x2 )
    x4 = Flatten()( x3 )
    encoder_output = Dense( s, activation='linear', use_bias=True, name='encoder_output')( x4 )
    encoder = Model( encoder_input, encoder_output, name='Encoder' )


    # Decoder
    decoder_input = Input( shape=(s), name='decoder_input')
    y1 = Dense( 4096, use_bias=True, activation='relu' )( decoder_input )
    y2 = Reshape( (8, 8, 64) )( y1 )
    y3 = Conv2DTranspose( 64, (3, 3), padding='same', strides=(2,2), use_bias=True, activation='relu' )( y2 )
    y4 = Conv2DTranspose(  1, (3, 3), padding='same', strides=(2,2), use_bias=True, activation='relu' )( y3 )
    y5 = Flatten()( y4 )
    decoder_output = Dense( data.shape[1], activation='linear' )( y5 )
    decoder = Model( decoder_input, decoder_output, name='Decoder' )

    print( encoder.summary() )
    print( decoder.summary() )


    # Auto Encoder
    model_input = encoder_input
    model_output = decoder( encoder_output )
    model = Model( model_input, model_output, name='Auto_Encoder' )

    model.compile( loss='mean_squared_error', optimizer='adam', metrics=['mse'] )
    history = model.fit( data, data, epochs=50, batch_size=10, verbose=2)


    # 学習結果の保存
    model.save( 'AE_model_s={}.h5'.format(s) )
    encoder.save( 'AE_encoder_s={}.h5'.format(s) )
    decoder.save( 'AE_decoder_s={}.h5'.format(s) )

    # 学習結果がある場合上の保存する3行をコメントアウトし、以下の3行のコメントアウトを外す
    #model   = tf.keras.models.load_model( 'AE_model.h5' )
    #encoder = tf.keras.models.load_model( 'AE_encoder.h5' )
    #decoder = tf.keras.models.load_model( 'AE_decoder.h5' )


    # 入力・出力画像の描画
    output = model.predict( data )   # 入力 data に対する AE の出力
    encode = encoder.predict( data ) # 入力 data に対する Encoder の出力

    # AE 出力画像の表示（100個）
    plt.figure( figsize=(8, 8))
    for k in range( 100 ):
        plt.subplot(10, 10, k+1) # 8x8 の subplot window の左上からk+1番目
        plt.xticks([])
        plt.yticks([])
        plt.imshow( output[k].reshape(64, 64), cmap='gray', interpolation='None' )
    plt.savefig("Outnput_images_s={}.png".format(s))


    # 2次元潜在変数の分布
    plt.figure( figsize=(8, 8))
    for k in range( encode.shape[0] ):
        plt.scatter( encode[k,0], encode[k,1], c='blue', s=3 )
        latent.append((encode[k,0], encode[k,1]))
    plt.savefig("latent_variable_s={}.png".format(s))

    plt.figure( figsize=(8,8))
    for k in range( 100 ):
        plt.scatter( encode[k,0], encode[k,1], c=color[label[k]], s=3 )
    plt.savefig("latent_variable_c_s={}.png".format(s))


# 2次元潜在変数の分布（カラー）
# 各数字(label)に基づき色を変更
plt.figure( figsize=(8,8))
markers1 = [",", "o", "v", "^", "*", "1", "X", "d"]
for i in range(2, 11):
    for k in range( 100 ):
        plt.scatter( latent[10 * (i - 2) + k][0], latent[10 * (i - 2) + k][1], c=color[label[k]], s=3, marker=markers1[i-2] )
plt.savefig("latent_variable_c.png")
