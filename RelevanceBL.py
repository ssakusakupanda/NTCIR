#-*- encoding: utf-8 -*-
import sys
import argparse
import fileinput
import json
import glob

from pyknp import Jumanpp
from pyknp import KNP
import codecs

import re
from collections import Counter

def format_text(text):

    '''
    MeCabに入れる前のツイートの整形方法例
    '''

    text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text=re.sub('RT', "", text)
    text=re.sub('お気に入り', "", text)
    text=re.sub('まとめ', "", text)
    text=re.sub(r'[!-~]', "", text)#半角記号,数字,英字
    text=re.sub(r'[︰-＠]', "", text)#全角記号
    text=re.sub('[\n、。「」]', " ", text)#改行文字

    return text

def main():

    Topic = []
    Utterance = []
    Relevance = []

    regex  = u'[^ぁ-ん]+'

    all_filepaths=glob.glob('./training/*')
    for filepath in all_filepaths:
        lines = [line.rstrip() for line in fileinput.input(
            filepath, openhook=fileinput.hook_encoded('utf-8'))]

        # JSON全体の文法チェック
        try:
            arguments = json.loads('\n'.join(lines))
        except json.JSONDecodeError as e:
            print('エラーあり')
            print(e)
            exit(1)

        # Display title
        #print(arguments[0]["Topic"])
        
        for argument in arguments:
            Topic.append(argument["Topic"])
            Utterance.append(argument["Utterance"])
            Relevance.append(argument["Relevance"])       

    TrueDataset = {}
    correctAnswer = 0
    for line in list(set(Utterance)): 
        T_List = [] 
        R_list = []
        for line_l in range(len(Utterance)):
            if line == Utterance[line_l]:
                T_List.append(Topic[line_l])
                R_list.append(Relevance[line_l])
        TrueDataset[Counter(T_List).most_common()[0][0] + ":" + line] = str(Counter(R_list).most_common()[0][0])

    # Analyze Utterance using Juman++ & knp
    jumanpp = Jumanpp()
    with open("incorrect.txt","w") as wf:
        line_cnt = len(TrueDataset)
        now_line_cnt = 0
        for key, label in TrueDataset.items():
            tpc,utr = key.split(":")[0],key.split(":")[1]

        #print(tpc + ":" + utr + "[" + label + "]")

        #parse Topic
            topic_analyed_List = []
            try:
                #0.7909880035111675
                #s = tpc.split("を")[-2] + "を" + tpc.split("を")[-1].split("べきである")[0] 
                #topic_result = jumanpp.analysis(s)
                topic_result = jumanpp.analysis(format_text(tpc))
                #print(s)
                for mrph in topic_result.mrph_list():
                    try :
                        if len(re.findall(regex, mrph.midasi)) > 0:
                            if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi): 
                                topic_analyed_List.append(mrph.midasi)
                    except:
                        continue
            except:
                #print("Error.",tpc)
                continue

        #parse Utterance
            utter_analyed_List = []
            try:
                utter_result = jumanpp.analysis(utr)
                for mrph in utter_result.mrph_list():
                    try :
                        if len(re.findall(regex, mrph.midasi)) > 0:
                            if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi): 
                                utter_analyed_List.append(mrph.midasi)
                    except:
                        continue
            except:
                #print("Error.",utr)
                continue

            #print((set(topic_analyed_List) & set(utter_analyed_List)),len(set(topic_analyed_List) & set(utter_analyed_List)))

            if (len(set(topic_analyed_List) & set(utter_analyed_List)) > 0):
                #print("1:",label)
                if int(label) == 1:
                    correctAnswer += 1
                else:
                    wf.write(tpc + ":" + utr + "[" + "1" + ":" +label + "]\n")
            else:
                #print("0:",label)
                if int(label) == 0:
                    correctAnswer += 1
                else:
                    wf.write(tpc + ":" + utr + "[" + "0" + ":" +label + "]\n")
            now_line_cnt += 1
            #print( now_line_cnt, "/", line_cnt)

    print("acurracy:",correctAnswer*1.0 / len(TrueDataset))

if __name__ == '__main__':
    main()
