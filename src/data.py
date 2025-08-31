import csv
import pickle
import numpy as np
from math import exp
from tqdm import tqdm

trainFiles = ["chessData.csv", "tactic_evals.csv"]
TO_DUMP = 3000000

def convertCentipawnsToProb(centipawns: int) -> float:
    # https://lichess.org/page/accuracy
    # https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/eval.scala
    #(-1, +1)
    return 2 / (1 + exp(-0.00368208 * centipawns)) - 1
    #(0, 100)
    #return 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
piece_to_channel = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,   #white
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  #black
}
def convertFENtoBitBoard(fen: str) -> tuple[np.ndarray, int]:
    bitboard = np.zeros((12, 8, 8), dtype=np.uint8)
    data = fen.split()
    board = data[0]
    pers = 1
    if data[1] == 'b':
        board = board.swapcase()
        pers = -1
    ranks = board.split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit(): file_idx += int(char)
            else:
                channel = piece_to_channel[char]
                bitboard[channel, 7 - rank_idx, file_idx] = 1
                file_idx += 1
    return (bitboard, pers)
def dumpToFile(f, file):
    reader = csv.reader(f)
    next(reader)
    toDumpBoard = []
    toDumpEval = []
    negativeEval = 0
    postiveEval = 0
    zeroEval = 0
    for row in tqdm(reader):
        board, pers = convertFENtoBitBoard(row[0])
        num = int(row[1].strip('#')) * pers
        evl = convertCentipawnsToProb(num)
        if num == 0: zeroEval += 1
        elif num < 0: negativeEval += 1
        else: postiveEval += 1
        toDumpBoard.append(board)
        toDumpEval.append(evl)
        if(len(toDumpEval) > TO_DUMP):
            pickle.dump([
                np.stack(toDumpBoard),
                np.array(toDumpEval)],
                file)
            toDumpBoard = []
            toDumpEval = []
    print(f"+ve: {postiveEval}\n-ve: {negativeEval}\nze: {zeroEval}\ntot: {postiveEval+negativeEval+zeroEval}")
    pickle.dump([
        np.stack(toDumpBoard),
        np.array(toDumpEval)],
        file)

if __name__ == "__main__":
    print("generating train.bin")
    file = open("data/train.bin", "wb")
    for tfile in trainFiles:
        with open("data/"+tfile, "r") as f: dumpToFile(f, file)
    file.close()

    print("generating test.bin")
    file = open("data/test.bin", "wb")
    with open("data/random_evals.csv", "r") as f: dumpToFile(f, file)
    file.close()
