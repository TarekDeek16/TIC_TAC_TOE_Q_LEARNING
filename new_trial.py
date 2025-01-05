import itertools
import random


def is_valid_state(board):
    x_count = sum(row.count("X") for row in board)
    o_count = sum(row.count("O") for row in board)

    # moves of X should be equal or more by 1 than moves of O, if not, not valid
    if not (x_count == o_count or (x_count - o_count) == 1):
        return False

    # if both of them has won, not valid since the game stops when one wins
    if has_won(board, "X") and has_won(board, "O"):
        return False

    # if x has won -> number of moves pf x should be more by 1, if not, not valid
    if has_won(board, "X") and x_count != o_count + 1:
        return False

    # if y has won -> number of moves of y should be equal to moves of x
    if has_won(board, "O") and x_count != o_count:
        return False

    # if the game has not ended yet or has ended in a valid state
    return True


def has_won(board, player):
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


def is_draw(board):
    # check if all cells are full, if not -> not draw
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                return False

    # if all cells are full and someone has won -> not draw
    if has_won(board, "X") or has_won(board, "O"):
        return False
    return True


def give_reward(board):
    if has_won(board, "O"):
        return 1
    elif has_won(board, "X"):
        return -1
    elif is_draw(board):
        return 0.5
    return 0


def generate_states_rewards_table():
    states_rewards_table = {}  # a dictionary to store state(string) -> reward(double 0->1)
    all_states = itertools.product("XO ", repeat=9)  # 3 elements : X, O, Empty in 9 positions, 3^9 tuples
    for state in all_states:
        board = [list(state[:3]), list(state[3:6]), list(state[6:])]
        if is_valid_state(board):
            states_rewards_table["".join(state)] = give_reward(board)
    return states_rewards_table


def initialize_q_table(states_rewards_table):
    q_table_function = {}
    for state in states_rewards_table.keys():
        # q-values for each possible actions: O, otherwise : -2
        q_table_function[state] = [0.0] * 9
        for i in range(9):
            if state[i] != " ":
                q_table_function[state][i] = -2
    return q_table_function


alpha = 0.7  # learning rate
gamma = 0.9  # discount factor, how important is the maximum q-value in the next state
epsilon = 0.2  # exploration rate -> chooses random move


# exploitation rate = (1-epsilon) -> it chooses the best move from the q-table

def do_action(state, player):
    valid_actions = [i for i in range(9) if state[i] == " "]  # all valid actions(empty cells)

    if random.uniform(0, 1) < epsilon:  # exploration
        best_action = random.choice(valid_actions)
    else:  # exploitation: choose action with the highest q-value from valid actions
        highest_q_value = max(q_table[state])
        best_actions = [i for i, q in enumerate(q_table[state]) if q == highest_q_value]
        best_action = random.choice(best_actions)  # index of where it is

    # best action: stores the index of the best position to play

    next_state = list(state)  # change it to list to apply the next move (list 0->9)
    next_state[best_action] = player  # applying the move
    next_state = "".join(next_state)  # returning back the state to string after applying the move
    # best action played

    return best_action, next_state
    # return the index of the action to update its q-value Q(state,action)
    # return the next state to calculate the next Q-value


def update_q_value(state, action, reward, next_state):
    if next_state in q_table:
        max_future_q = max(q_table[next_state])
    else:
        raise Exception("Unexpected Error")

    return q_table[state][action] + alpha * (reward + gamma * max_future_q - q_table[state][action])


q_table = initialize_q_table(generate_states_rewards_table())
states = generate_states_rewards_table()


def play_game(is_training):
    global epsilon, q_table, states
    state = "         "  # start with an empty board
    current_player = "X"

    def get_board(s):
        return [list(s[:3]), list(s[3:6]), list(s[6:])]

    while state in states.keys():
        board = get_board(state)

        # check for game over
        if has_won(board, "X") or has_won(board, "O") or is_draw(board):
            break

        if current_player == "O":  # O turn, computer turn
            action, next_state = do_action(state, current_player)
            if next_state in states:
                reward = states[next_state]
                new_q_value = update_q_value(state, action, reward, next_state)
                q_table[state][action] = new_q_value
                state = next_state
            else:
                break
        else:  # X turn
            valid_actions = [i for i in range(9) if state[i] == " "]

            if is_training:
                pos = random.choice(valid_actions)
            else:
                # opponent play manually, print board
                for i in range(3):
                    row = state[3 * i:3 * (i + 1)]
                    print(f"{row[0]}|{row[1]}|{row[2]}".replace(" ", "_"))

                pos = int(input('Enter your move (0-8): '))

            # apply the move
            state = state[:pos] + current_player + state[pos + 1:]

        current_player = "O" if current_player == "X" else "X"


def train():
    global epsilon
    for i in range(100000):
        play_game(is_training=True)
        epsilon = max(0.01, epsilon * (0.9999 ** i))  # decay epsilon over time

    epsilon = 0
    play_game(is_training=True)


def play():
    train()
    play_game(False)


play()
