import pygame, sys
pygame.init()

# global variables
WIDTH, HEIGHT = 512, 512
MESSAGE_MARGIN = 30
white = (255, 255, 255)
black = (0, 0, 0)
turn = 1
first_move = True
colors = [
    (60, 32, 16), # brown
    (16, 144, 85), # green
    (199, 69, 50), # red
    (229, 201, 33), # yellow
    (217, 121, 170), # pink
    (103, 55, 134), # purple
    (5, 107, 173), # blue
    (214, 116, 33) # orange
]
# color of grid on the board
board = [
    [ 7, 6, 5, 4, 3, 2, 1, 0 ],
    [ 2, 7, 4, 1, 6, 3, 0, 5 ],
    [ 1, 4, 7, 2, 5, 0, 3, 6 ],
    [ 4, 5, 6, 7, 0, 1, 2, 3 ],
    [ 3, 2, 1, 0, 7, 6, 5, 4 ],
    [ 6, 3, 0, 5, 2, 7, 4, 1 ],
    [ 5, 0, 3, 6, 1, 4, 7, 2 ],
    [ 0, 1, 2, 3, 4, 5, 6, 7 ]
]
# position of towers for black and white players
# e.g., tower[0][0] -> (row,  col) position of brown tower (=colors[0])
#                      of black player
tower = [
    [ (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0) ],
    [ (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7) ]
]
playercolor = [ black, white ]

font = pygame.font.SysFont("comicsans",20)

# create window
screen = pygame.display.set_mode((WIDTH, HEIGHT + MESSAGE_MARGIN))
pygame.display.set_caption("KAMISADO")

def draw_grid():
    for i in range(8): # index of row
        for j in range(8): # index of col
            # locate a rect at i-th row and j-th col
            rect = pygame.Rect(
                (j * WIDTH/8, i * HEIGHT/8), (WIDTH/8, HEIGHT/8))
            pygame.draw.rect(screen, colors[board[i][j]], rect, width=0)

def draw_tower():
    for p in range(2): # player (0 or 1)
        for i in range(8): # each tower
            center = (tower[p][i][1] * WIDTH/8 + WIDTH/16,
                        tower[p][i][0] * HEIGHT/8 + HEIGHT/16)
            pygame.draw.circle(screen, colors[i], center, WIDTH/16, width=0)
            pygame.draw.circle(screen, playercolor[p], center, WIDTH/16, width=2)

def _message_margin(txt, backgr, textcol):
    rect = pygame.Rect( (0, HEIGHT), (WIDTH, MESSAGE_MARGIN) )
    pygame.draw.rect(screen, backgr, rect)
    pos_text = font.render(txt, True, textcol)
    pos_rect = pos_text.get_rect()
    pos_rect.center = ( WIDTH/2, HEIGHT + MESSAGE_MARGIN/2 )
    screen.blit(pos_text, pos_rect)

def msg_turn():
    title = "Win" if win else "Turn"

    if turn == 0: # turn of blacks
        txt = "{}: {}".format(title, 'black')
        _message_margin(txt, black, white)
    else:
        txt = "{}: {}".format(title, 'white')
        _message_margin(txt, white, black)

def calc_grid(pos):
    # get row and col indices & return
    # note that y -> row number, x -> col number
    return (int(pos[1] / (HEIGHT/8)), int(pos[0] / (WIDTH/8)))

def check_valid_tower(turn, pos):
    global tower_to_move

    # for each tower of this turn,
    #print(tower[turn], pos)
    for i, t in enumerate(tower[turn]):
        # find the tower at the clicked cell
        if pos == t:
            tower_to_move = i
            return True
    return False

# examine if a tower in cur can move to tar
# 1) turn: a player can only move a tower forward 
#          in the direction from home row to opposite
# 2) cur, tar: it should be a straight line from cur to pos
#              that is, a column or diagonals
#              furthermore, no other tower is on the path
def check_valid_target(turn, cur, tar):
    if cur == tar:
        return False

    # row idx increases from cur to tar if turn == 0 (black)
    forward = 1 if turn == 0 else -1

    # check if tower is to move forward
    if (tar[0] - cur[0]) * forward < 0:
        return False

    #print("tar", tar, "cur", cur)
    
    # moving vertical (the same column)
    if tar[1] == cur[1]:
        c = tar[1]
        rd = 1 if tar[0] - cur[0] > 0 else -1
        rs = range(cur[0]+rd, tar[0]+rd, rd)
        # see if any tower is on the path
        for t in range(2): 
            for r in rs:
                if (r,c) in tower[t]:
                    return False
        return True

    # moving diagonal
    elif abs(tar[1] - cur[1]) == abs(tar[0] - cur[0]):
        rd = 1 if tar[0] - cur[0] > 0 else -1
        rs = range(cur[0]+rd, tar[0]+rd, rd)
        cd = 1 if tar[1] - cur[1] > 0 else -1
        cs = range(cur[1]+cd, tar[1]+cd, cd)
        for r, c in zip(rs, cs):
            for t in range(2):
                if (r, c) in tower[t]:
                    return False
        return True
    return False

def highlight_cell(turn, pos):
    rect = pygame.Rect(
                (pos[1] * WIDTH/8, pos[0] * HEIGHT/8),
                (WIDTH/8, HEIGHT/8))
    pygame.draw.rect(screen, playercolor[turn], rect, width=2)

def goal_reached(pos):
    if turn == 0 and pos[0] == 7:
        return True
    elif turn == 1 and pos[0] == 0:
        return True
    return False

def is_blocked():
    forward = 1 if turn == 0 else -1
    towers = tower[0] + tower[1]
    if tower_to_move_pos[1] == 0: # left most column
        t1 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1])
        t2 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1]+1)
        if (t1 in towers) and (t2 in towers):
            return True
    elif tower_to_move_pos[1] == 7: # right most column
        t1 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1])
        t2 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1]-1)
        if (t1 in towers) and (t2 in towers):
            return True
    else:
        t1 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1])
        t2 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1]-1)
        t3 = (tower_to_move_pos[0]+forward, tower_to_move_pos[1]+1)
        if (t1 in towers) and (t2 in towers) and (t3 in towers):
            return True
    return False

def msg_blocked():
    if turn == 0: # turn of blacks
        txt = "Blocked: next turn -> {} in 10 secs".format('white')
        _message_margin(txt, black, white)
    else:
        txt = "Blocked: next turn -> {} in 10 secs".format('black')
        _message_margin(txt, white, black)

# nowhere to move (blocked), then the current player (A) gives up the turn
# and the opposite (B) takes the turn with the tower of the color
# of the cell of A's last tower
def next_move_for_blocked():
    global turn, tower_to_move, tower_to_move_pos
    turn = 1 if turn == 0 else 0
    tower_to_move = board[tower_to_move_pos[0]][tower_to_move_pos[1]]
    tower_to_move_pos = tower[turn][tower_to_move]

win = False
running = True
tower_to_move = -1
tower_to_move_pos = None
target_cell = None
last_player_before_blocked = turn
blocked_pos_list = []

if __name__ == "__main__":
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not win:
                # calculate the clicked piece
                pos = calc_grid(pygame.mouse.get_pos())

                # if tower to move is not yet selected
                if tower_to_move < 0:
                    if check_valid_tower(turn, pos):
                        #print('tower_to_move set', idx, tower_to_move)
                        tower_to_move_pos = pos

                # if tower to move is selected but not the target cell 
                elif tower_to_move >=0 and target_cell is None:
                    # toggle to reset tower_to_move
                    # the first selection of the first move only can be
                    # toggled
                    if first_move and tower_to_move_pos == pos:
                        #print('tower_to_move reset')
                        tower_to_move = -1
                    
                    # examine if moving to the target cell is valid
                    elif check_valid_target(turn, tower_to_move_pos, pos):
                        #print('valid move')

                        # with every valid move, blocked_pos_list is reset
                        blocked_pos_list = [ pos ]
                        last_player_before_blocked = turn

                        # move the tower
                        tower[turn][tower_to_move] = pos

                        if goal_reached(pos):
                            win = True
                            tower_to_move_pos = pos
                        else:
                            # switch turn
                            turn = 1 if turn == 0 else 0
                            
                            # next turn's tower is the tower with the color
                            # identical to the background color to which the tower 
                            # moved at the last turn by the opposite player
                            tower_to_move = board[pos[0]][pos[1]]
                            tower_to_move_pos = tower[turn][tower_to_move]
                            first_move = False


        # display current state on the board
        draw_grid()
        draw_tower()
        msg_turn()

        # highlight the tower to move or the final move won the game
        if tower_to_move >=0 and target_cell is None:
            highlight_cell(turn, tower_to_move_pos)

        # checking if blocked tower without any trigger such as mouse clicks
        if not win and tower_to_move >=0 and is_blocked():
            # show message for 10 sec
            msg_blocked()
            pygame.display.update()
            pygame.time.delay(10000)

            if tower_to_move_pos not in blocked_pos_list:
                # increase dealock count for checking deadlock type 2
                blocked_pos_list.append(tower_to_move_pos)

                # swith turn and the tower to move in the next turn 
                next_move_for_blocked()
            
            # if the next tower is already in the blocked tower list
            # that is, deadlock
            else:
                # game is over and the opposit player wins
                tower_to_move = -1
                turn = 1 if last_player_before_blocked == 0 else 0
                win = True

            # every event during the above delay is ignored
            pygame.event.clear()
            continue

        pygame.display.update()
        pygame.time.delay(200)

    pygame.quit()
    sys.exit()