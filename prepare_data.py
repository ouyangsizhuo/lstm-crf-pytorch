from xml.dom.minidom import parse
from bs4 import BeautifulSoup
import en_core_web_sm
from collections import defaultdict
import os

def read_label(label_file: str):
    label_dict = defaultdict(set)
    with open('./prepare_data/data_offset.txt') as f:
        for line in f:
            l = line.strip().split('\t')
            label_dict[l[0]].add((l[1], l[2], l[3]))
    return label_dict

def read_file():
    path1 ='./train_xml'
    path2 ='./tab'
    file_name = []
    name_tab = []
    os.listdir(path1)
    os.listdir(path2)
    for file in os.listdir(path1):
            name_tab.append(os.path.join(path2, file[:-4]+'.tab'))
            file_name.append(os.path.join(path1, file))
    return file_name,name_tab

def readXML():
    file_name,name_tab = read_file()
    for i in range(len(file_name)):
        domTree = parse(file_name[i])
        rootNode = domTree.documentElement
        file = open('./prepare_data/data_offset.txt', mode='w')
        Mentions = rootNode.getElementsByTagName("Mention")
        for Mention in Mentions:
            if Mention.hasAttribute("start"):
                if Mention.hasAttribute("str"):
                    if Mention.hasAttribute("type"):
                        if Mention.hasAttribute("section"):
                            file.write("{0}\t{1}\t{2}\t{3}\n".format(str(Mention.getAttribute("section")),
                                                                     str(Mention.getAttribute("start")),
                                                                     str(Mention.getAttribute("str")),
                                                                     str(Mention.getAttribute("type"))))
        file.close()

        soup = BeautifulSoup(open(file_name[i]), 'xml')
        id_list = []
        string_list = []
        id_string = {}
        for index in soup.find_all('Section'):
            id_list.append(index.attrs['id'])
            string_list.append(index.string)
            id_string = dict(zip(id_list,string_list))
        label_dict = read_label('./prepare_data/data_offset.txt')
        for id in id_list:
            lable,text = splitWord(str(id_string[id]), label_dict[id])
            input = open(name_tab[i], 'a+',newline='\n')
            write_file(lable, text,input)

def splitWord(sentences, label_set):
    nlp = en_core_web_sm.load()
    doc = nlp(sentences)
    word_start = {}
    i = []
    text = []
    for token in doc:
        i.append(token.i)
    lable = ['O']*(max(i)+1)

    for token in doc:
        word_start[token.idx] = token.text
        text.append(token.text)
        for annotation in label_set:
            offset, token_, label = annotation
            if ',' in str(offset):
                continue
            if int(token.idx) == int(offset) :
                for idx,word in enumerate(token_.split(' ')):
                    if idx == 0:
                        lable[token.i] = 'B-{0}'.format(label)
                    else:
                        lable[token.i+idx] = 'I-{0}'.format(label)
            else:
                    pass
        else:
            pass
    return lable,text

def write_file(lable,text,input):
    f = open('./prepare_data/word_tag_test.txt', mode='w')
    for i in range(len(text)):
        f.write("{0}\t{1}\n".format(text[i],lable[i]))
    f.close()
    with open('./prepare_data/word_tag_test.txt') as ff:
        for line in ff:
            l=line.strip().split('\t')
            if len(l) == 1:
                del(l)
                continue
            if (l[0] =='*')|(l[0] =='<')|(l[0] =='>')|(l[0] =='=')|(l[0] =='/'):
                del(l)
                continue
            else:
                input.write("{0}\t{1}\n".format(l[0],l[1]))
    ff.close()

def train_test_data():
    path1 = './tab/train'
    path2 = './tab/valid'
    path3 = './tab/test'

    train_filename = []
    valid_filename = []
    test_filename = []

    for name in os.listdir(path1):
        train_filename.append(os.path.join(path1, name))
    for name in os.listdir(path2):
        valid_filename.append(os.path.join(path2, name))
    for name in os.listdir(path3):
        test_filename.append(os.path.join(path3, name))
    return train_filename,valid_filename,test_filename



def prepare(input,filename):
    for name in filename:
        clean = open('./prepare_data/clean.txt', 'w')
        with open(name, 'r') as file:
            for lines in file:
                line = lines.strip().split('\t')
                if len(line) != 2:
                    clean.write('\n')
                    continue
                if line[0] == '.':
                    clean.write('{0}/{1}'.format(line[0], line[-1]))
                    clean.write('\n')
                else:
                    clean.write('{0}/{1} '.format(line[0], line[-1]))
            clean.close()
            file.close()
        with open('./prepare_data/clean.txt', 'r') as f:
            for ls in f:
                l = list(ls)
                input.write("".join(l))
            f.close()
    input.close()

if __name__=='__main__':
    #readXML()
    train = open('./prepare_data/train.txt','w',newline='\n')
    valid = open('./prepare_data/valid.txt','w',newline='\n')
    test = open('./prepare_data/test.txt','w',newline='\n')
    train_filename ,valid_filename , test_filename = train_test_data()
    prepare(train,train_filename)
    prepare(valid,valid_filename)
    prepare(test,test_filename)
