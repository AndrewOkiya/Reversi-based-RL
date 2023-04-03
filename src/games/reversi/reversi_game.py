import numpy as np
from src.games.game import Game
from src.games.reversi.reversi_logic import ReversiLogic

class ReversiGame(Game):
    """
    定义一个游戏类的接口，其他各类游戏可实现它
    """

    def __init__(self, n=8):
        self.n = n  # 棋盘大小 n*n
        self.logic = ReversiLogic(self.n)

    def init(self, board=None):
        """使用棋盘矩阵初始化"""
        if board is None:
            self.logic = ReversiLogic(self.n)
        else:
            self.logic.set_pieces(board)
        return self.logic.pieces

    def display(self, board):
        """打印当前棋盘状态"""
        self.init(board=board)
        self.logic.display()

    def get_board_size(self):
        return self.n, self.n

    def get_action_size(self):
        """获取动作总数，其中 self.n ** 2 为走棋，剩下一个为无路可走"""
        return self.n ** 2 + 1

    def get_winner(self, board):
        """获取游戏是否结束等"""
        self.init(board=board)

        if len(self.logic.get_legal_moves(1)):  # 玩家 1 可走
            return self.WinnerState.GAME_RUNNING
        if len(self.logic.get_legal_moves(-1)):  # 玩家 -1 可走
            return self.WinnerState.GAME_RUNNING

        player1_count = self.logic.count(1)  # player1 得分
        player2_count = self.logic.count(-1)  # player2 得分
        # 比较两个玩家判断哪个赢
        if player1_count == player2_count:  # 平局
            return self.WinnerState.DRAW
        elif player1_count > player2_count:  # player1 胜利
            return self.WinnerState.PLAYER1_WIN
        else:
            return self.WinnerState.PLAYER2_WIN

    def get_legal_moves(self, player, board):
        """获取行动力矩阵"""
        self.init(board=board)
        legal_moves = self.logic.get_legal_moves(player)
        res = np.zeros(self.get_action_size(), dtype=np.int)
        if len(legal_moves) == 0:
            # 无路可走的情况，这里 res[-1] 刚好是 res[self.n ** 2]
            res[-1] = 1
        for x, y in legal_moves:
            res[x * self.n + y] = 1
        return res

    def get_current_state(self):
        """获取棋盘当前状态"""
        return self.logic.pieces

    def evaluate(self,board, validmove):
        ans=np.zeros(5,dtype=np.float_) #星位差，角差，边缘子比，子数比，行动力比
        ft=[0,0,0]
        f=[0,0,0]
        i,j,k,fn,fo =0,0,0,0,0
        dir= [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),(0, 1)]
        for i in range(self.n):
            for j in range(self.n):
                if (board[i][j] == 0):
                    if validmove[1][i*self.n+j]:
                        fn+=1
                    if validmove[0][i*self.n+j]:
                        fo+=1
                else:
                    for k in range(8):
                        tem1=i+dir[k][0]
                        tem2=j+dir[k][1]
                        if 0 <= tem1 < self.n and 0 <= tem2 < self.n:
                            if(board[tem1][tem2]==0):
                                ft[board[i][j]+1]+=1
                                break
                    f[board[i][j]+1]+=1
        if (f[2] > f[0]):
            ans[3] = (f[0]) / (f[2] + f[0])
        elif (f[2] < f[0]):
            ans[3] = -(f[0]) / (f[2] + f[0])
        if (ft[2] > ft[0]):
            ans[2] = -(ft[0]) / (ft[2]+ft[0])
        elif (ft[2] < ft[0]):
            ans[2] = (ft[0]) / (ft[2]+ft[0])
        if (fn > fo):
            ans[4] = (fn) / (fn + fo)
        elif (fn < fo):
            ans[4] = -(fo) / (fn + fo)
        if board[0][0]:
            ans[1]+=board[0][0]
        else:
            ans[0] +=board[1][0]
            ans[0] +=board[0][1]
            ans[0] +=board[1][1]
        if board[self.n-1][self.n-1]:
            ans[1] +=board[self.n-1][self.n-1]
        else:
            ans[0] +=board[self.n-2][self.n-1]
            ans[0] +=board[self.n-1][self.n-2]
            ans[0] +=board[self.n-2][self.n-2]
        if board[0][self.n-1]:
            ans[1] +=board[0][self.n-1]
        else:
            ans[0] += board[0][self.n-2]
            ans[0] += board[1][self.n-1]
            ans[0] += board[1][self.n-2]
        if board[self.n-1][0]:
            ans[1] +=board[self.n-1][0]
        else:
            ans[0] +=board[self.n-1][1]
            ans[0] +=board[self.n-2][0]
            ans[0] +=board[self.n-2][1]
        ans[0]/=4
        ans[1]/=4
        return ans
    
    def get_relative_state(self, player, board):
        """获取相对矩阵"""
        return player * board

    def get_next_state(self, player, action, board,turn=0):
        """玩家 player 执行 action 后的棋盘状态"""
        self.init(board=board)
        if 0 <= action < self.n ** 2:
            self.logic.execute_move((action // self.n, action % self.n), player)
            turn+=1
        return self.logic.pieces, -player,turn

    def get_others(self,player):
        board=self.get_relative_state(player,self.get_current_state())
        moves=[self.get_legal_moves(-1,board),self.get_legal_moves(1,board)]
        return np.append(self.evaluate(board,moves),player)

    def get_symmetries(self, board, others,pi):
        # pi[:-1] 代表去除最后一个 action
        pi_board = np.reshape(pi[:-1], self.get_board_size())
        res = []
        for i in range(4):
            """注意：这里旋转与翻转 rot90 和 fliplr 都类似于视图，即其所关联的对象内容一改全改"""
            # 旋转
            new_board = np.rot90(board, i)
            new_pi = np.rot90(pi_board, i)
            res += [(new_board, others,list(new_pi.ravel()) + [pi[-1]])]

            # 翻转
            new_board = np.fliplr(new_board)
            new_pi = np.fliplr(new_pi)
            res += [(new_board, others,list(new_pi.ravel()) + [pi[-1]])]
        return res


if __name__ == "__main__":
    a = ReversiGame(8)
    a.init([[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
    b = a.get_symmetries(a.get_current_state(), np.zeros(a.get_action_size()))
    # b = a.get_legal_moves(1)
    # print(b.reshape(1, -1))
    # player = 1
    # while True:
    #     a.display()
    #     print(a.board.get_legal_moves(player))
    #     x, y = map(int, input().split())
    #     tmp, player,cache = a.get_next_state(player, x * 8 + y)
    #     pass
    # print(a.get_legal_moves(1))
    # a.display()
