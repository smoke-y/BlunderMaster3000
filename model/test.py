import torch
import pickle
import numpy as np
from model import ChessEval

def convertBitBoardToFEN(bitboard: np.ndarray) -> str:
    channel_to_piece = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',   # white
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'  # black
    }
    fen_rows = []
    for rank in range(7, -1, -1):
        fen_row = ""
        empty_count = 0
        for file in range(8):
            piece_found = False
            for channel in range(12):
                if bitboard[channel, rank, file] == 1:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += channel_to_piece[channel]
                    piece_found = True
                    break
            if not piece_found: empty_count += 1
        if empty_count > 0: fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen_board = '/'.join(fen_rows)
    fen = f"{fen_board} w KQkq - 0 1"
    return fen

device = "cuda"

model = ChessEval()
model.load_state_dict(torch.load("eval.pth", weights_only=True))
model.to(device)

file = open("data/test.bin", "rb")
while True:
    try:
        bigBatch = pickle.load(file)
        boards = torch.from_numpy(bigBatch[0])
        evls = torch.tensor(bigBatch[1]).unsqueeze(-1)
        for i in range(0, len(boards)-1):
            b = boards[i:i+1, :, :, :].type(torch.float32).to(device)
            e = evls[i:i+1, :].type(torch.float32).to(device)
            pred = model.forward(b)
            pred = pred.detach().cpu().numpy()
            e = e.detach().cpu().numpy()
            if (e > 0 and pred < 0) or (e < 0 and pred > 0):
                b = b.detach().cpu().squeeze(0).numpy()
                print(convertBitBoardToFEN(b), end=" | ")
            print(f"pred: {pred}, eval: {e}")
    except EOFError: break
file.close()
