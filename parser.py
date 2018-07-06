import pandas as pd
import re

class parser():
    def __init__(self):
        self.FILENAME = ''
        self.PARAMS = []
        self.DATA = []

    def parse_data(self, file_in):
        self.FILENAME = file_in
        with open(self.FILENAME, 'r') as fin:
            data = fin.read().splitlines(True)
            i = 0
            headEnd = 0
            footStart = 0
            inHeader = True
            while inHeader:
                i += 1
                line = data[i]
                if line.startswith(";"):
                    continue
                else:
                    headEnd = i;
                    PARAMS = data[i-1].split()
                    PARAMS.pop(0)
                    PARAMS = self.check_headers(PARAMS)
                    inHeader = False

            inFooter = False
            while not inFooter:
                i += 1
                line = data[i]
                if line.startswith(";"):
                    footStart= i;
                    inFooter = True
                else:
                    continue

            output = data[headEnd:(footStart-1)]

        with open('temp.dat', 'w') as fout:
            fout.writelines(output)

        df = pd.read_table('temp.dat', sep="\s+",names=PARAMS,usecols=PARAMS)
        self.DATA =df

    def check_headers(self, PARAMS):

        parse_dict = {
            'PosCmd#00[0]': 'xcommand',
            'PosCmd#01[0]': 'ycommand',
            'PosFbk#00[0]': 'xfeedback',
            'PosFbk#01[0]': 'yfeedback',
            'CurFbk#00[0]': 'xcurrent',
            'CurFbk#01[0]': 'ycurrent',
        }

        newparam = [ parse_dict.get(item,item) for item in PARAMS ]
        return newparam

    def get_x(self):
        X = self.DATA[['xcommand','xfeedback']]
        return X

    def get_y(self):
        Y=self.DATA[['ycommand','yfeedback']]
        return Y

    def get_all_com_fbk(self):
        comfbk = self.DATA[['xcommand','ycommand','xfeedback','yfeedback']]
        return comfbk

    def get_all(self):
        all = self.DATA
        return all
