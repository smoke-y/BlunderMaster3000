import torch
import pickle
from model import ChessEval
from sys import argv

device = "cuda"
EPOCH = 5
MINI_BATCH = 34

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
model = ChessEval()
if len(argv) > 1:
    print("loading model...")
    model.load_state_dict(torch.load("eval.pth", weights_only=True))
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
model = torch.compile(model)

file = open("data/train.bin", "rb")
epoch = 0
batchCount = 0
avgLoss = 0
while epoch < EPOCH:
    print("EPOCH:", epoch)
    while True:
        try:
            bigBatch = pickle.load(file)
            boards = torch.from_numpy(bigBatch[0])
            evls = torch.from_numpy(bigBatch[1]).unsqueeze(-1)
            turns = torch.from_numpy(bigBatch[2]).unsqueeze(-1)
            for i in range(0, len(boards), MINI_BATCH):
                endx = min(i+MINI_BATCH, len(boards))
                b = boards[i:endx, :, :, :].type(torch.float32).to(device)
                e = evls[i:endx, :].type(torch.float32).to(device)
                t = turns[i:endx, :].type(torch.float32).to(device)
                optimizer.zero_grad()
                pred = model.forward(b)
                loss = criterion(pred, e)
                loss.backward()
                optimizer.step()
                lossVal = loss.detach().cpu()
                avgLoss += lossVal
                batchCount += 1
                #print(f"loss: {lossVal}", end='\r')
                #print(f"loss: {lossVal}")
        except EOFError:
            epoch += 1
            print(f"avg loss: {avgLoss/batchCount}")
            file.seek(0)
            batchCount = 0
            avgLoss = 0.0
            torch.save(model._orig_mod.state_dict(), "eval.pth")
        except:
            epoch = EPOCH
            break
file.close()

print("writing weights...")
torch.save(model._orig_mod.state_dict(), "eval.pth")
