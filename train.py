from model import *
from utils import *
from evaluate import *
from dataloader import *

def load_data():
    data = dataloader()
    batch = []
    cti = load_tkn_to_idx('./prepare_data/train.txt.char_to_idx') # char_to_idx
    wti = load_tkn_to_idx('./prepare_data/train.txt.word_to_idx') # word_to_idx
    itt = load_idx_to_tkn('./prepare_data/train.txt.tag_to_idx') # idx_to_tkn
    print("loading %s..." % './prepare_data/train.txt.csv')
    with open('./prepare_data/train.txt.csv') as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        for line in block.split("\n"):
            x, y = line.split("\t")
            x = [x.split(":") for x in x.split(" ")]
            y = [int(y)] if HRE else [int(x) for x in y.split(" ")]
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append_item(xc = xc, xw = xw, y0 = y)
        data.append_row()
    data.strip()
    for _batch in data.split():
        xc, xw, y0, lens = _batch.sort()
        xc, xw = data.tensor(xc, xw, lens)
        _, y0 = data.tensor(None, y0, sos = True)
        batch.append((xc, xw, y0))
    print("data size: %d" % len(data.y0))
    print("batch size: %d" % BATCH_SIZE)
    return batch, cti, wti, itt

def train():
    num_epochs = 20
    batch, cti, wti, itt = load_data()
    model = rnn_crf(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint('model', model) if isfile('model') else 0
    filename = re.sub("\.epoch[0-9]+$", "", 'model')
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y0 in batch:
            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(batch)
        print('loss: {0}'.format(loss_sum))
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt]
            evaluate(predict('./prepare_data/valid.txt', *args), True)
            model.train()
            print()

if __name__ == "__main__":
    train()
