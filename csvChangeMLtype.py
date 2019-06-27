#-*- encoding: utf-8 -*-
import csv
import re

with open('data2.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    with open('AllArgumentSetJumanpp.csv',  newline='') as f:
        dataReader = csv.reader(f)
        for line in dataReader:
            try:
                #line_l[1] : Relevance, line_l[2]:FactChecking, line_l[3]:Stance
                checkCandidate = line_l[1]
                line_l = line[0].split("\t")
                if re.match('\d',checkCandidate):
                    writer.writerow([checkCandidate, line_l[0]])
            except:
                continue
