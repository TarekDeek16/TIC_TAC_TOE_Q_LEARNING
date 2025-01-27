import numpy as np
import random
import tkinter as tk
from tkinter import messagebox


class TicTacToeGUI:
    def __init__(self):
        self.window = tk.Tk()  # create window
        self.window.title("Tic Tac Toe")  # title of the window
        self.buttons = []  # array of buttons that represents the board
        self.clicked_position = None  # last played(clicked) move
        self.create_board()  # initialize buttons array

    def create_board(self):
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(
                    self.window,
                    text="",
                    font=('Arial', 20),
                    width=5,
                    height=2,
                    # when clicked, call button_clicked method passing to it the indices of clicked button
                    command=lambda row=i, col=j: self.button_click(row, col)
                )
                button.grid(row=i, column=j)
                row.append(button)
            self.buttons.append(row)

    def button_click(self, row, col):
        self.clicked_position = row * 3 + col + 1
        # equation of the number of played(clicked) move assigned to clicked position when clicked

    def update_board(self, board):
        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                self.buttons[i][j]['text'] = board[index]

    @staticmethod
    def show_message(message):
        messagebox.showinfo("Game Over", message)

    def reset_board(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]['text'] = ""
        self.clicked_position = None

    def get_move(self):
        self.clicked_position = None
        while self.clicked_position is None:  # keep waiting for input
            self.window.update()  # to keep it responsive and allowing it to handle clicks
        return self.clicked_position


class TicTacToe:
    def __init__(self, playerX, playerO):  # this is a tic-tac-toe class that takes 2 players (agent or random or human)
        #                                  #  the goal of the class is to make games between players based on the tic-tac-toe rules
        self.board = [' '] * 9
        self.playerX, self.playerO = playerX, playerO
        self.playerX_turn = True
        self.gui = None

    def set_gui(self, GUI):
        self.gui = GUI

    def play_game(self):
        self.playerO.start_game()
        while True:
            if self.playerX_turn:
                player, turn, opponent = self.playerX, 'X', self.playerO
            else:
                player, turn, opponent = self.playerO, 'O', self.playerX
            position = player.move(self.board)  # played move position
            if self.board[position - 1] != ' ':  # illegal move
                self.gui.show_message("Illegal move!")
                break
            self.board[position - 1] = turn  # play the move, space: position, index = position - 1

            if self.gui:
                self.gui.update_board(self.board)

            if player.breed == "Qlearner" and opponent.breed == "Qlearner":  # during training
                if self.has_won(turn):  # if it has won, give winner a point and loser a minus point
                    player.reward(1, self.board)
                    opponent.reward(-1, self.board)
                    break
                if self.board_full():  # tie game
                    player.reward(0.5, self.board)
                    opponent.reward(0.5, self.board)
                    break
                opponent.reward(0, self.board)  # neither has own, nor tie =>continue(0 reward)

            if self.has_won(turn):
                self.gui.show_message("!!You have Lost!!")
                break
            if self.board_full():
                self.gui.show_message("!!  Tie  !!")
                break

            self.playerX_turn = not self.playerX_turn  # switch turns

    def has_won(self, player):
        board = [self.board[0:3], self.board[3:6], self.board[6:9]]
        for i in range(3):
            # rows
            if all(board[i][j] == player for j in range(3)):
                return True
            # columns
            if all(board[j][i] == player for j in range(3)):  # col fixed, iterate over row ->repeat
                return True

        if all(board[i][i] == player for i in range(3)):  # pos+ diagonal
            return True
        if all(board[i][2 - i] == player for i in range(3)):  # neg- diagonal
            return True

        return False

    def board_full(self):
        return ' ' not in self.board  # not in board, means board does not contain space => full



class Player:
    def __init__(self, gui=None):
        self.breed = "human"
        self.gui = gui

    def move(self, board):
        return self.gui.get_move()

    @staticmethod
    def available_moves(board):
        return [i + 1 for i in range(9) if board[i] == ' ']


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
        self.breed = "random"


    def move(self, board):
        return random.choice(self.available_moves(board))


class Q_learn_agent(Player):
    def __init__(self, ):
        super().__init__()
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q_table = {}  # (state, action) tuples work as keys: Q values
        self.epsilon = 0.2  # exploration(random moves) factor (probability)
        self.alpha = 0.3  # learning rate
        self.gamma = 0.9  # discount factor for future rewards

    def start_game(self):
        self.current_state = (' ',) * 9
        self.last_move = None

    def getQ(self, state, action):  # state as a tuple
        # encourage exploration, by setting the initial to 1 (called optimistic initial value)
        # ** q value of a bad action will be reduced by a negative reward, so, not be chosen in the future
        if self.q_table.get((state, action)) is None:
            self.q_table[(state, action)] = 1.0
        return self.q_table.get((state, action))

    def move(self, board):
        self.current_state = tuple(board)
        actions = self.available_moves(board)

        if random.random() < self.epsilon:  # explore!
            self.last_move = random.choice(actions)
            return self.last_move

        qs = [self.getQ(self.current_state, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than action with the same q score, choose among them randomly  // this aligns perfectly with optimistic q vlue to encourage exploration
            best_actions = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_actions)  # index of the random chosen action among them
        else:
            i = qs.index(maxQ)  # one with highest Q score, return its index

        self.last_move = actions[i]  # this gives the action by the index in the available actions array
        return actions[i]

    def reward(self, value, board):
        if self.last_move:
            self.learn(self.current_state, self.last_move, value, tuple(board))

    def learn(self, state, action, reward, next_state):
        current_q = self.getQ(state, action)
        max_new_q = max([self.getQ(next_state, a) for a in self.available_moves(state)])
        self.q_table[(state, action)] = current_q + self.alpha * ((reward + self.gamma * max_new_q) - current_q)


if __name__ == "__main__":
    # make 2 players to us them for training
    p1 = Q_learn_agent()
    p2 = Q_learn_agent()

    print("Training AI...")
    for i in range(200000):
        t = TicTacToe(p1, p2)
        t.play_game()
    print(p1.q_table)
    print("Training has completed. Starting game...")
    p2.epsilon = 0  # set exploration to 0 after training, to NOT CHOOSE RANDOMLY

    # start the game with human player with GUI
    gui = TicTacToeGUI()
    human = Player(gui)  # pass GUI to human player
    while True:
        game = TicTacToe(human, p2)
        game.set_gui(gui)
        game.play_game()
        if not messagebox.askyesno("Play Again?", "Another Game?"):
            break
        game.board = [' '] * 9
        gui.reset_board()
