__author__ = 'PC-LiNing'

import numpy
import official_score


word_embedding_size = 200
num_classes = 19
PF_dim = 30
final_mebedding_size = word_embedding_size + 2*PF_dim
MAX_DOCUMENT_LENGTH = 50

# [-49,48]
# PF_embeddings = numpy.random.uniform(low=-0.5,high=0.5,size=(98, PF_dim))
PF_embeddings = numpy.load('PF_embeddings.npy')


# [-49,48]
# n is sentence length , max_length is the max length.
# return [max_length,2*PF_dim]
def  get_PF(e1,e2,n,max_length):
    pos_embeddings = numpy.zeros(shape=(max_length, 2*PF_dim),dtype=numpy.float32)
    vec = range(n)
    pos = 0
    for i in vec:
        p1 = i - e1 + 49
        p2 = i - e2 + 49
        if p1 < 0 or p1 > 97:
            d1 = numpy.zeros(shape=(PF_dim,), dtype=numpy.float32)
        else:
            d1 = PF_embeddings[p1]
        if p2 < 0 or p2 > 97:
            d2 = numpy.zeros(shape=(PF_dim,), dtype=numpy.float32)
        else:
            d2 = PF_embeddings[p2]
        pos_vec = numpy.concatenate((d1,d2),axis=0)
        pos_embeddings[pos] = pos_vec
        pos += 1
    return pos_embeddings


# (sent,type,weight,entity)
def  SemEval_data(data_file):
     file = open(data_file)
     sentence=[]
     label=[]
     sdp = []
     entity = []
     i=1
     for line in file.readlines():
         if i % 5 == 1:
             sentence.append(line.replace('\n',''))
         if i % 5 == 3:
             label.append(official_score.transfer_label(line.replace('\n','')))
         if i % 5 == 4:
             entitys = line.replace('\n','').split(',')
             entity.append([int(item) for item in entitys])
         if i % 5 == 0:
             if line.replace('\n','') == '':
                 sdp.append([])
             else:
                 words = line.replace('\n','').split(',')
                 sdp.append([int(word) for word in words])
         i+=1
     # parse
     train_data=[]
     for i in range(0,len(sentence)):
         sen=sentence[i]
         type=label[i]
         weight=sdp[i]
         et=entity[i]
         train_data.append((sen,type,weight,et))
     return train_data


# (max_length,)
# init 1.2 , weight = 3.0
def getWeightVector(n,max_length,weight,entitys):
    vec = numpy.zeros(max_length,dtype=numpy.float32)
    for i in range(n):
        vec[i] = 0.7
    for pos in weight:
        vec[pos] = 2.5
    return vec


# parse SemEval train data
# (sent,type,weight)
def load_train_data():
    # numpy.save('PF_embeddings.npy',PF_embeddings)
    semeval_data = SemEval_data("Dep_train_8000_50.txt")
    Train_Size = len(semeval_data)
    train_pf = numpy.ndarray(shape=(Train_Size, MAX_DOCUMENT_LENGTH, 2*PF_dim),dtype=numpy.float32)
    train_weight = numpy.ndarray(shape=(Train_Size,MAX_DOCUMENT_LENGTH),dtype=numpy.float32)
    i = 0
    for one in semeval_data:
        sentence = one[0]
        length = len(sentence.split(' '))
        entitys = one[3]
        # [max_length,pos_embed]
        pos_embed = get_PF(entitys[0],entitys[1],length,MAX_DOCUMENT_LENGTH)
        train_pf[i] = pos_embed
        train_weight[i]=getWeightVector(length,MAX_DOCUMENT_LENGTH,one[2],entitys)
        i+=1

    return train_pf, train_weight

# parse SemEval test data
def load_test_data():
    semeval_data = SemEval_data("Dep_test_2717_50.txt")
    Train_Size = len(semeval_data)
    train_pf = numpy.ndarray(shape=(Train_Size, MAX_DOCUMENT_LENGTH, 2*PF_dim),dtype=numpy.float32)
    train_weight = numpy.ndarray(shape=(Train_Size,MAX_DOCUMENT_LENGTH),dtype=numpy.float32)
    i = 0
    for one in semeval_data:
        sentence = one[0]
        length = len(sentence.split(' '))
        entitys = one[3]
        # [max_length,pos_embed]
        pos_embed = get_PF(entitys[0],entitys[1],length,MAX_DOCUMENT_LENGTH)
        train_pf[i] = pos_embed
        train_weight[i]=getWeightVector(length,MAX_DOCUMENT_LENGTH,one[2],entitys)
        i+=1

    return train_pf,train_weight

