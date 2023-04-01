import os
import time
from src.games.nnet_agent import NeuralNetAgent
import tensorflow as tf
# from tensorflow.python.keras.layers import *
# from tensorflow.python.keras.models import *
# from tensorflow.python.keras.optimizers import *
# from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import TensorBoard
import numpy as np


class NNetWrapper(NeuralNetAgent):
    def __init__(self, game, args):
        self.args = args
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, others_state, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        others_state =np.asarray(others_state)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        others_state=others_state[:,np.newaxis, :]
        tb_call_back = TensorBoard(log_dir=os.path.join(self.args.logs_folder, str(time.time())),  # log 目录
                                   histogram_freq=0,
                                   batch_size=self.args.batch_size,
                                   write_graph=True,  # 是否存储网络结构图
                                   write_grads=True,  # 是否可视化梯度直方图
                                   write_images=True,  # 是否可视化参数
                                   embeddings_freq=0,
                                   embeddings_layer_names=None,
                                   embeddings_metadata=None)

        # 使用当前的棋盘和其他状态作输入，拟合 (可行点的概率，权值)
        self.nnet.model.fit(x=np.concatenate((input_boards, others_state), axis=1), y=[target_pis, target_vs], batch_size=self.args.batch_size,
                            epochs=self.args.epochs,
                            callbacks=[tb_call_back])

    def predict(self, board, others):
        """
        board: np array with board
        """
        # preparing input
        if self.args.use_tpu:
            # 使用 TPU 时因为有 8 个核心，所以这里的大小必须是 8 的整数倍 QAQ，没找到其他办法
            tmp = []
            for i in range(8):
                tmp.append(board)
            board = np.array(tmp)
        else:
            board = board[np.newaxis, :, :]
        others=others[np.newaxis,np.newaxis, :]
        # run
        pi, v = self.nnet.model.predict(np.concatenate((board, others), axis=1))

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        self.nnet.model.save_weights(filepath)
        # if self.args.use_tpu:
        #     # 使用 TPU 时需先将 model sync 到 cpu 中再存储（这里还有 bug）
        #     model_tmp = self.nnet.model.sync_to_cpu()
        #     model_tmp.save_weights(filepath)
        # else:
        #     self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)


class OthelloNNet(object):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args
        # Neural Net
        self.input_boards = Input(shape=(self.board_x+1, self.board_y))
        board = Lambda(lambda x: x[:, :8, :8])(self.input_boards)
        status_param = Lambda(lambda x: x[:, 8, :7])(self.input_boards)
        x_image = Reshape((self.board_x, self.board_y, 1))(board) # batch_size  x board_x x board_y x 1
        params = Reshape((7,))(status_param)
        h_conv1 = Activation('relu')(Conv2D(args.num_channels, 3, padding='same', use_bias=True)(x_image)) # batch_size  x board_x x board_y x num_channels
        board=Add()([h_conv1,board])
        h_conv4_flat = Flatten()(board)
        inputfc=Concatenate(axis=-1)([h_conv4_flat, params])
        s_fc1 = Dropout(args.dropout)(Activation('relu')(Dense(5, use_bias=False)(inputfc)))  # batch_size x 1024# batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc1)  # batch_size x 1
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

        if args.use_tpu:
            # 支持 TPU 训练，关键代码在这里，需要一个TPU
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
                )
            )
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=tf.optimizers.Adam(learning_rate=args.lr))
