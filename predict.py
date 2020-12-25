from model import *
from utils import *
from dataloader import *

def load_model(args):
    cti = load_tkn_to_idx('./prepare_data/train.txt.char_to_idx') # char_to_idx
    wti = load_tkn_to_idx('./prepare_data/train.txt.word_to_idx') # word_to_idx
    itt = load_idx_to_tkn('./prepare_data/train.txt.tag_to_idx') # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    """
    cti:包括出现的所有字符（字母，标点，特殊符号，<PAD>,<SOS>,<EOS>,<UNK>等）
    wti:包括train.txt中分词后出现的所有单词
    itt:包括所有的标签
    """
    print(model)
    load_checkpoint('model.epoch20', model)
    return model, cti, wti, itt

def run_model(model, data, itt):
    with torch.no_grad():
        model.eval()
        for batch in data.split():
            xc, xw, _, lens = batch.sort()
            xc, xw = data.tensor(xc, xw, lens)
            y1 = model.decode(xc, xw, lens)
            batch.y1 = [[itt[i] for i in x] for x in y1]
            batch.unsort()
            for x0, y0, y1 in zip(batch.x0, batch.y0, batch.y1):
                if not HRE:
                    y0, y1 = [y0], [y1]
                for x0, y0, y1 in zip(x0, y0, y1):
                    yield x0, y0, y1

def predict(model, cti, wti, itt, filename):
    data = dataloader()
    with open(filename) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        for x0 in block.split("\n"):
            if re.match("\S+/\S+( \S+/\S+)*$", x0): # word/tag
                x0, y0 = zip(*[re.split("/(?=[^/]+$)", x) for x in x0.split(" ")])
                x1 = list(map(normalize, x0))
            else:
                y0 = []
                if re.match("[^\t]+\t\S+$", x0): # sentence \t label
                    x0, *y0 = x0.split("\t")
                x0 = tokenize(x0)
                x1 = list(map(normalize, x0))
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0, x1, xc, xw, y0)
        data.append_row()
    data.strip()
    return run_model(model, data, itt)

if __name__ == "__main__":
    result = predict('./prepare_data/test.txt', *load_model())
    func = tag_to_txt if TASK else lambda *x: x
    test_out = open('test_out.tab', 'w',newline='\n')
    for x0, y0, y1 in result:
       for i in range(len(x0)):
            test_out.write('{0}\t{1}\t{2}\n'.format(x0[i],y0[i],y1[i]))
    test_out.close()
