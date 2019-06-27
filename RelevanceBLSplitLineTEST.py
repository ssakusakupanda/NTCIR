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
    text=re.sub(r'[!-~]', "", text)#半角記号,数字,英字
    text=re.sub(r'[︰-＠]', "", text)#全角記号
    text=re.sub('[\n、。「」]', " ", text)#改行文字

    return text

def main():

    Topic = []
    Utterance = []
    Relevance = []
    ID  = []
    
    regex  = u'[^ぁ-ん]+'

    #学習用データ form[label, Topic & Utterce]
    #wf_Data = open("Tpc&UTRtEST.csv","w")

    all_filepaths=glob.glob('./testGS/*')
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
            ID.append(argument["ID"])
            Topic.append(argument["Topic"])
            Utterance.append(argument["Utterance"])
            Relevance.append(argument["Relevance"])       

    TrueDataset = {}
    correctAnswer_0 = 0
    correctAnswer_1 = 0
    incorrectAnswer_0 = 0
    incorrectAnswer_1 = 0
    
    for line in list(set(Utterance)):
        T_List = [] 
        R_list = []
        id_tag = 0
        for line_l in range(len(Utterance)):
            if line == Utterance[line_l]:
                T_List.append(Topic[line_l])
                R_list.append(Relevance[line_l])
                id_tag = ID[line_l]
        TrueDataset[Counter(T_List).most_common()[0][0] + ":" + line + ":" + id_tag ] = str(Counter(R_list).most_common()[0][0])

    sorted(TrueDataset.items())


    # Analyze Utterance using Juman++ & knp
    jumanpp = Jumanpp()
    with open("CommonWords.csv","w") as wf:
        wf.write("label,A,B\n")
        line_cnt = len(TrueDataset)
        now_line_cnt = 0
        for key, label in TrueDataset.items():
            tpc,utr,id = key.split(":")[0],key.split(":")[1],key.split(":")[2]
            topANDutrANDlabelList = []

            #parse Topic
            topic_analyed_List = []
            topANDutrANDlabelList.append("Topic")
            try:
                #0.7909880035111675
                #s = tpc.split("を")[-2] + "を" + tpc.split("を")[-1].split("べきである")[0] 
                #topic_result = jumanpp.analysis(s)
                topic_result = jumanpp.analysis(format_text(tpc))
                #print(s)
                for mrph in topic_result.mrph_list():
                    try :
                        if len(re.findall(regex, mrph.genkei)) > 0:
                            if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi):
                                if "数量" in mrph.imis:
                                    topic_analyed_List.append(mrph.genkei)
                                    topANDutrANDlabelList.append("[数]") 
                                else:
                                    topic_analyed_List.append(mrph.genkei)
                                    topANDutrANDlabelList.append(mrph.genkei)
                    except:
                        continue
            except:
                continue

        #parse Utterance
            utter_analyed_List = []
            topANDutrANDlabelList.append("Utterance")
            try:
                if "、" in utr:
                    utrList = utr.split("、")
                    for sentence in utrList:

                        #reigi
                        if sentence == "":
                            continue
                        
                        utter_result = jumanpp.analysis(sentence)
                        for mrph in utter_result.mrph_list():
                            try :
                                if len(re.findall(regex, mrph.genkei)) > 0:
                                    if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi):
                                        if "数量" in mrph.imis:
                                            utter_analyed_List.append(mrph.genkei)
                                            topANDutrANDlabelList.append("[数]") 
                                        else:
                                            utter_analyed_List.append(mrph.genkei)
                                            topANDutrANDlabelList.append(mrph.genkei)

                                else:
                                    continue
                            except:
                                print("error")
                                continue

                else:
                    utter_result = jumanpp.analysis(utr)
                    for mrph in utter_result.mrph_list():
                        try :
                            if len(re.findall(regex, mrph.genkei)) > 0:
                                if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi):
                                    if "数量" in mrph.imis:
                                        utter_analyed_List.append(mrph.genkei)
                                        topANDutrANDlabelList.append("[数]") 
                                    else:
                                        utter_analyed_List.append(mrph.genkei)
                                        topANDutrANDlabelList.append(mrph.genkei)
                        except:
                            print("error")
                            continue
                topANDutrANDlabelList.append("END")
                    
            except:
                print("error")
                continue

            #if "END" in topANDutrANDlabelList:
                #wf_Data.write(str(label) + "," + " ".join(topANDutrANDlabelList[:-1]) + "\n")#+ " [---] " + "{\"ID\":\"" + id + "\",\"Topic\":\"" + tpc + "\",\"Utterance\":\"" + utr + "\",\"Relevance\":\"" + "null" + "\",\"Fact-checkability\":null,\"Stance\":null,\"Class\":null}\n")
#            if "END" in topANDutrANDlabelList:
#                wf_Data.write(str(label) + "," + " ".join(topANDutrANDlabelList[:-1])+ " [---] " + "{\"ID\":\"" + id + "\",\"Topic\":\"" + tpc + "\",\"Utterance\":\"" + utr + "\",\"Relevance\":\"" + "null" + "\",\"Fact-checkability\":null,\"Stance\":null,\"Class\":null}\n")


            if (len(set(topic_analyed_List) & set(utter_analyed_List)) > 0):
                #wf.write("{\"ID\":\"" + id + "\",\"Topic\":\"" + tpc + "\",\"Utterance\":\"" + utr + "\",\"Relevance\":\"" + "1" + "\",\"Fact-checkability\":null,\"Stance\":null,\"Class\":null},\n")
                wf.write(str(label) + ",1," + str(1) + "\n")
            else:
                wf.write(str(label) + ",1," + str(0) + "\n")
                #wf.write("{\"ID\":\"" + id + "\",\"Topic\":\"" + tpc + "\",\"Utterance\":\"" + utr + "\",\"Relevance\":\"" + "0" + "\",\"Fact-checkability\":null,\"Stance\":null,\"Class\":null},\n")


#            if (len(set(topic_analyed_List) & set(utter_analyed_List)) > 0):
#                if int(label) == 1:
#                    correctAnswer_1 += 1
#                else:
#                    incorrectAnswer_1 += 1
#            
#            else:
#                if int(label) == 0:
#                    correctAnswer_0 += 1
#                else:
#                    incorrectAnswer_0 += 1

            now_line_cnt += 1
            print(now_line_cnt,len(TrueDataset),line_cnt)

        correctAnswer = correctAnswer_0 + correctAnswer_1
        print(correctAnswer*1.0/now_line_cnt, " ans0:",correctAnswer_0," ans1:", correctAnswer_1, " miss:", now_line_cnt - correctAnswer)
        print( "詳細:", "p0t0", correctAnswer_0, "p0t1", incorrectAnswer_0, "p1t0", incorrectAnswer_1, "p1t1", correctAnswer_1,)

    label_cnt = 0
    for text,label in TrueDataset.items():
        if int(label) == 1:
            label_cnt += 1
    print(label_cnt/len(TrueDataset))


            #print(now_line_cnt)
#print(correctAnswer_0, correctAnswer_1, (correctAnswer_0 +correctAnswer_1)*1.0 / len(TrueDataset))

if __name__ == '__main__':
    main()
