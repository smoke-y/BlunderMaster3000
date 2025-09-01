import chess

MAXVAL = 10000
values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

def value(b):
    if b.is_game_over():
      if b.result() == "1-0":
        return MAXVAL
      elif b.result() == "0-1":
        return -MAXVAL
      else:
        return 0

    val = 0.0
    # piece values
    pm = b.piece_map()
    for x in pm:
      tval = values[pm[x].piece_type]
      if pm[x].color == chess.WHITE:
        val += tval
      else:
        val -= tval

    # add a number of legal moves term
    bak = b.turn
    b.turn = chess.WHITE
    val += 0.1 * b.legal_moves.count()
    b.turn = chess.BLACK
    val -= 0.1 * b.legal_moves.count()
    b.turn = bak

    return val

def computer_minimax(board, depth, a, b, big=False):
  if depth >= 5 or board.is_game_over():
    return value(board)
  # white is maximizing player
  turn = board.turn
  if turn == chess.WHITE:
    ret = -MAXVAL
  else:
    ret = MAXVAL
  if big:
    bret = []

  # can prune here with beam search
  isort = []
  for e in board.legal_moves:
    board.push(e)
    isort.append((value(board), e))
    board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=board.turn)

  # beam search beyond depth 3
  if depth >= 3:
    move = move[:10]

  for e in [x[1] for x in move]:
    board.push(e)
    tval = computer_minimax(board, depth+1, a, b)
    board.pop()
    if big:
      bret.append((tval, e))
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b:
        break  # b cut-off
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b:
        break  # a cut-off
  if big:
    return ret, bret
  else:
    return ret

def explore_leaves(board):
  cval, ret = computer_minimax(board, 0, a=-MAXVAL, b=MAXVAL, big=True)
  return ret

def computer_move(board):
  # computer move
  move = sorted(explore_leaves(board), key=lambda x: x[0], reverse=board.turn)
  if len(move) == 0:
    return
  print("top 3:")
  for i,m in enumerate(move[0:3]):
    print("  ",m)
  print(board.turn, "moving", move[0][1])
  board.push(move[0][1])
