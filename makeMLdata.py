#-*- encoding: utf-8 -*-
import sys
import argparse
import fileinput
import json
import glob

from pyknp import Jumanpp
from pyknp import KNP
import codecs

from collections import Counter


def main():

    all_filepaths=glob.glob('./training/*')
    #print("frhifr",all_filepaths)

    Topic     = []
    Utterance = []
    Relevance = []
    FactCheck = []
    Stance    = []

    for filepath in all_filepaths:

        # args = get_args()
        # JSON読み込み
        # src = '-' if not hasattr(args, 'json_file') else args.json_file

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
            FactCheck.append(argument["Fact-checkability"])
            Stance.append(argument["Stance"])            

    TrueDataset = []
    for line in list(set(Utterance)):  
        cnt = 0
        R_list = []
        F_list = []
        S_list = []
        for line_l in range(len(Utterance)):
            if line == Utterance[line_l]:
                cnt += 1
                R_list.append(Relevance[line_l])
                F_list.append(FactCheck[line_l])
                S_list.append(Stance[line_l])
        plane = line + " " + str(Counter(R_list).most_common()[0][0]) + " " + str(Counter(F_list).most_common()[0][0]) + " " + str(Counter(S_list).most_common()[0][0])
        if not ( (cnt == 5 and Counter(S_list).most_common()[0][1] == 2) or (cnt == 3 and Counter(S_list).most_common()[0][1] == 1) ):
            TrueDataset.append(plane)

    # Analyze Utterance using Juman++
    jumanpp = Jumanpp()
    for arguments in TrueDataset:
        #print(argument["Utterance"],argument["Relevance"],argument["Fact-checkability"],argument["Stance"],argument["Class"])
        argument = arguments.split(" ")
        result = jumanpp.analysis(argument[0])
        analyed_argument = ""
        for mrph in result.mrph_list():
            if ("名詞" in mrph.hinsi or "動詞" in mrph.hinsi): 
                analyed_argument += mrph.midasi + " "


        analyed_argument += "\t"
        analyed_argument += argument[1] + "\t"
        analyed_argument += argument[2] + "\t"
        analyed_argument += argument[3]
        
        print(analyed_argument)


if __name__ == '__main__':
    main()
