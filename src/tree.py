import chess
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ChessEval

BATCH = 64
TOP = 3

class Node:
    def __init__(self, board: str = ""):
        self.children = []
        self.board = board

device = "cuda"
model = ChessEval()
#model.load_state_dict(torch.load("eval.pth", weights_only=True))
model.to(device)

piece_to_channel = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,   #white
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  #black
}
def convertFENtoBitBoard(board: str) -> np.ndarray:
    bitboard = np.zeros((12, 8, 8), dtype=np.uint8)
    ranks = board.split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit(): file_idx += int(char)
            else:
                channel = piece_to_channel[char]
                bitboard[channel, 7 - rank_idx, file_idx] = 1
                file_idx += 1
    return bitboard

def infer(fens: list) -> list:
    data = fens[0].split()
    boards = []
    toPlay = data[1]
    if toPlay == 'b': boards = [
            convertFENtoBitBoard(fen.split()[0].swapcase())
            for fen in fens]
    else: boards = [
            convertFENtoBitBoard(fen.split()[0])
            for fen in fens]
    boards = np.stack(boards)
    boards = torch.from_numpy(boards).type(torch.float32).to(device)
    pred = model.forward(boards).flatten()
    topk = min(TOP, pred.numel())
    return torch.topk(pred, k=topk, dim=-1).indices.tolist()

def buildTree(board: chess.Board, depth: int) -> Node:
    root = Node(board.fen())
    if depth == 0: return root
    legalMoves = []
    moves = []
    for legalMove in board.legal_moves:
        board.push(legalMove)
        legalMoves.append(board.fen())
        board.pop()
    for i in range(0, len(legalMoves), BATCH):
        endx = min(len(legalMoves), i+BATCH)
        tops = infer(legalMoves[i:endx])
        for t in tops: moves.append(legalMoves[i+t])
    for move in moves:
        childBoard = chess.Board(move)
        childNode = buildTree(childBoard, depth-1)
        root.children.append(childNode)
    return root

def dumpTree(root: Node, pad: int=0) -> None:
    print(f"{pad*"    "}{root.board}")
    for child in root.children: dumpTree(child, pad+1)

root = buildTree(chess.Board(), 2)
dumpTree(root)
