__author__ = 'PC-LiNing'

import re
import codecs
import os

label_str = {
    0:'Cause-Effect(e1,e2)',
    1:'Cause-Effect(e2,e1)',
    2:'Instrument-Agency(e1,e2)',
    3:'Instrument-Agency(e2,e1)',
    4:'Product-Producer(e1,e2)',
    5:'Product-Producer(e2,e1)',
    6:'Content-Container(e1,e2)',
    7:'Content-Container(e2,e1)',
    8:'Entity-Origin(e1,e2)',
    9:'Entity-Origin(e2,e1)',
    10:'Entity-Destination(e1,e2)',
    11:'Entity-Destination(e2,e1)',
    12:'Component-Whole(e1,e2)',
    13:'Component-Whole(e2,e1)',
    14:'Member-Collection(e1,e2)',
    15:'Member-Collection(e2,e1)',
    16:'Message-Topic(e1,e2)',
    17:'Message-Topic(e2,e1)',
    18:'Other'

}


# relation label to number
def transfer_label(label):
    if label.startswith('Cause-Effect(e1,e2)'):
        return 0
    if label.startswith('Cause-Effect(e2,e1)'):
        return 1
    if label.startswith('Instrument-Agency(e1,e2)'):
        return 2
    if label.startswith('Instrument-Agency(e2,e1)'):
        return 3
    if label.startswith('Product-Producer(e1,e2)'):
        return 4
    if label.startswith('Product-Producer(e2,e1)'):
        return 5
    if label.startswith('Content-Container(e1,e2)'):
        return 6
    if label.startswith('Content-Container(e2,e1)'):
        return 7
    if label.startswith('Entity-Origin(e1,e2)'):
        return 8
    if label.startswith('Entity-Origin(e2,e1)'):
        return 9
    if label.startswith('Entity-Destination(e1,e2)'):
        return 10
    if label.startswith('Entity-Destination(e2,e1)'):
        return 11
    if label.startswith('Component-Whole(e1,e2)'):
        return 12
    if label.startswith('Component-Whole(e2,e1)'):
        return 13
    if label.startswith('Member-Collection(e1,e2)'):
        return 14
    if label.startswith('Member-Collection(e2,e1)'):
        return 15
    if label.startswith('Message-Topic(e1,e2)'):
        return 16
    if label.startswith('Message-Topic(e2,e1)'):
        return 17
    if label.startswith('Other'):
        return 18


# official score is the macro-averaged F1-score for (9+1)-way classification,with directionality taken into account.
def official_score(y_predicts):
    # generate proposed file
    proposed_file = codecs.open("proposed_test_2717.txt","a",encoding='utf-8')
    i =1
    for item in y_predicts:
        proposed_file.write(str(i)+'\t'+label_str[item]+'\n')
        i += 1
    proposed_file.close()

    # perl script
    perl_script = './semeval2010_task8_scorer-v1.2_test.pl'
    predicts_file = 'proposed_test_2717.txt'
    content = os.popen("perl "+perl_script+" "+predicts_file).read()
    print(content.strip('\n'))
    # delete proposed file
    # os.remove(predicts_file)
    result = re.search(r'F1 = ([-+]?[0-9]*\.?[0-9]+)',content).group(0)
    return float(result.split()[2])


"""
file = open('proposed_test_labelfile.txt')
labels = []
for line in file.readlines():
    labels.append(transfer_label(line.strip('\n').split('\t')[1]))

print(official_score(labels))
"""








