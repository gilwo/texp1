#!/usr/bin/env python3

import xlrd
import csv
from itertools import groupby
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import argparse
import os

import pdb

# requirement:
# xlsxwriter
# pandas
# numpy
# matplotlib
# argparse

# def read_data_file(filename: str) -> pd.DataFrame:
#     data = pd.read_excel(filename, sheet_name=None) # read all sheets
#     if type(data) is dict:
#         return data['Sheet1']
#     return data

# columns name mapping
PART       = 'Participat'
TRIAL      = 'Trial'
TIME       = 'Time'
TARGET     = 'Target'
FILL1      = 'Filler 1'
COMP       = 'Competitor'
FILL2      = 'Filler 2'
TYPE       = 'Type'
VER        = 'Version'
# calculated special
FILLAvg    = "Filler Average"
NTargtAvg  = "Non Target Average"
TARGET_COLOR='red'
COMP_COLOR='blue'
NTargetCOLOR='green'
color_dict={COMP: COMP_COLOR, TARGET: TARGET_COLOR, NTargtAvg: NTargetCOLOR, FILLAvg: NTargetCOLOR}
# Types
A          = 'a'
As         = 'as'
Am         = 'am'
AF         = 'af'
B          = 'b'
C          = 'c'
D          = 'd'
E          = 'e'
F          = 'f'
P          = 'p'
# cut off times
EARLY_TRIM = 200
TARGET_WORD_TIME = 2700
LATE_TRIM_1 = 3500 # for normal sentences
LATE_TRIM_2 = 4500 # for combined sentenced (trial type f or af)



# output_name=args.input.replace('.xlsx', '_result.xlsx')

#pdb.set_trace()

def get_data_raw(input) -> pd.DataFrame:
    data = pd.read_excel(input, sheet_name=None) # read all sheets
    if 'Sheet1' not in data:
        print("Sheet1 not found in excel sheets, geting the first sheet\n")
        workSet = data[list(data.keys())[0]].copy(deep=True)
    else:
        workSet = data['Sheet1'].copy(deep=True)

    return workSet



# # trim to normal sentences
# # normal work set
# nws = workSet[(workSet.Time >= EARLY_TRIM) &
#                              (workSet.Time <= LATE_TRIM_1) &
#                              (workSet.Type != 'f') &
#                              (workSet.Type != 'af')]

# read the message report and extract TOUCH_TARGET time stamp and stimuli timestamp,
# calculate ofset for TOUCH_TARGET from mimimum of stimuli (when multiple stimuli exists)
def get_message_report_ofs(input) -> pd.DataFrame:

    # input = "060919/old_message_report.xlsx"
    # input = args.input
    _d = pd.read_excel(input, sheet_name=None)
    _data = _d['all_trial_messages']
    
    # filter messages which are not TOUCH_TARGET nor stimuli ...
    _data2 = _data.loc[(_data['CURRENT_MSG_TEXT'] == 'TOUCH_TARGET') | (
        _data['CURRENT_MSG_TEXT'].str.contains('Stimuli'))]
    
    _data2[_data2['CURRENT_MSG_TEXT'].str.contains('Stimu')].pivot_table(
        index=['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], values='CURRENT_MSG_TIME', aggfunc=min)
    
    # find which trials are missing entries of stimuli touch report bundle
    _data3 = _data2.pivot_table(index=['RECORDING_SESSION_LABEL'], values=[
                                'CURRENT_MSG_TEXT'], columns=['TRIAL_INDEX'], aggfunc=[len])
    
    # rename value from 'Stimuli: <NAME>' to just 'Stimuli'
    _data4 = _data2.replace({'Stimuli:.*': 'Stimuli'}, regex=True)
    
    # pivot table to have columns for TOUCH_TARGET and Stimuli, have min value of CURRENT_MSG_TIME as the new value
    _data5 = _data4.pivot_table(index=['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], columns=[
        'CURRENT_MSG_TEXT'], values='CURRENT_MSG_TIME', aggfunc=min)
    
    # find out missing TOUCH_TARGET trials
    _data5[_data5['TOUCH_TARGET'].isnull()]
    
    # get the int value instead of float
    #_data6 = _data5[~_data5['TOUCH_TARGET'].isnull()].astype(int)
    
    # fill Nan as 0, set type to int and calculate offset from Stimuli to TOUCH_TARGET
    _data6 = _data5.fillna(0).astype(int).assign(
        TOUCH_FIXED=lambda x: x['TOUCH_TARGET'] - x['Stimuli'])
    
    # missing TOUCH_TARGET message report
    # _data6[_data6['TOUCH_FIXED'] < 0]
    
    # rename only the indexs 
    _data7 = _data6.rename_axis([PART, TRIAL])

    return _data7

# extract pairs of start and end offsets when TARGET is at value of 10 for each trial 
def get_10_ranges(workSet):

    #based on stack overflow: 
    # https://stackoverflow.com/questions/24281936/delimiting-contiguous-regions-with-values-above-a-certain-threshold-in-pandas-da/24283319#24283319

    _q = workSet.pivot_table(index=[PART, TRIAL, TIME],
                         values=[TARGET, COMP, FILL1, FILL2, TYPE],
                         aggfunc=lambda x: x)

    _q['tag'] = _q[TARGET] == 10
    fst = _q.index[_q['tag'] & ~ _q['tag'].shift(1).fillna(False)]
    lst = _q.index[_q['tag'] & ~ _q['tag'].shift(-1).fillna(False)]
    # maybe change the condition ?
    pr = [(i,j) for i,j in zip(fst, lst) if j > i]

    # return touple of ((PART,TRIAL,BEGIN), (PART,TRIAL,END))
    return pr


def find_cutoff_for_10(workSet):

    pr = get_10_ranges(workSet)

    # build a dataframe
    prdf = pd.DataFrame(columns=[PART, TRIAL, 'begin', 'end'])
    for i in pr:
        prdf = prdf.append({PART: i[0][0], TRIAL: i[0][1], 'begin': i[0][2], 'end': i[1][2]})

    prdf1 = prdf.set_index([PART, TRIAL])

    cutoff_ofs = get_message_report_ofs()

    pd.merge(prdf1, cutoff_ofs, on=[PART, TRIAL])


def plot_part_trial(data2, _p, _t):
    (_p, _t) = (411, 9)
    p = data2.loc[_p].loc[_t].plot(x=TIME, y=TARGET)
    p2 = p.bar(x=data2.loc[_p].loc[_t].TOUCH_FIXED.values[0],
               height=10, width=20, color="red")
    p3 = p.bar(x=1500, height=10, width=20, color='magenta')
    p4 = p.bar(x=2700, height=10, width=20, color='black')
    p.set_title("Part: {}, Trial: {}, Type: {}".format(
        _p, _t, data2.loc[_p].loc[_t].index[0]))
    p.legend(['Target', 'cutoff {}'.format(
        data2.loc[_p].loc[_t].TOUCH_FIXED.values[0]), 'qend: 1500', 'target: 2700'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
    parser.add_argument("indata", type=str, help="input data excel file")
    parser.add_argument("intouch", type=str, help="input touch data excel file")
    parser.add_argument("-o", dest="outdata", type=str, help="output data excel file")
    parser.add_argument("-t", dest="title", type=str, help="graph title prefix")
    args = parser.parse_args()

    data = get_data_raw(args.indata)
    touch_data = get_message_report_ofs(args.intouch)

    # merge data with touch point
    data_with_touch = pd.merge(data.set_index([PART, TRIAL]),
                               touch_data,
                               on=[PART, TRIAL]).\
        reset_index([PART, TRIAL]).\
        set_index([PART, TRIAL, TYPE])

    # if we want to plot target for specific participant and trial
    # plot_part_trial(data_with_touch, 411, 5)


    # move type from index to column
    _d = data_with_touch.reset_index(TYPE)

    _d = _d.assign(**{
        NTargtAvg: (_d[COMP]+_d[FILL1]+_d[FILL2]) / 3,
        FILLAvg: (_d[FILL1]+_d[FILL2])/2
    })

    # padd from touch point
    _d.loc[_d[TIME] >= _d['TOUCH_FIXED'], TARGET] = 10
    
    if args.outdata is not None:
        _d.to_excel(args.outdata)

    # chop time for  A(A, Am, As), B, C, D, E  (from 200 to 3500)
    _d2 = _d[(_d[TIME] >= 200) & (_d[TIME] <= 3500)]
    # # plot graphs
    # for t in [B, C, D, E]:
    #     _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
    #                                   values=[TARGET, COMP, FILL1, FILL2],
    #                                   aggfunc=np.average).\
    #         plot().set_title("{}type {}".format(args.title, t))
        


    #     pp.draw()

    # _d2[(_d2[TYPE] == A) | (_d2[TYPE] == Am) | (_d2[TYPE] == As)].pivot_table(index=[TIME],
    #                               values=[TARGET, COMP, FILL1, FILL2],
    #                               aggfunc=np.average).\
    #     plot().set_title("{}type {}".format(args.title, "A, Am, As"))
    # pp.draw()
    
    # chop time for  AF, F  (from 200)
    _d3 = _d[(_d[TIME] >= 200)]
    # # plot graphs
    # for t in [F, AF]:
    #     _d3[_d3[TYPE] == t].pivot_table(index=[TIME],
    #                                   values=[TARGET, COMP, FILL1, FILL2],
    #                                   aggfunc=np.average).\
    #         plot().set_title("{}type {}".format(args.title, t))
    #     pp.draw()

    for t in [B, D]:
        # _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
        #                                 values=[TARGET, NTargtAvg],
        #                                 aggfunc=np.average).\
        #     plot().set_title("{}type {}, non target".format(args.title, t))
        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TARGET, NTargtAvg],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.bar(x=1500, height=10, width=10, color='black')
        _p.bar(x=2700, height=10, width=10, color='black')
        # _p.legend([TARGET, NTargtAvg, 'qend: 1500', 'target: 2700'])
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'])
        _p.set_title("{}type {}, non target".format(args.title, t))

        pp.draw()

    # _d2[(_d2[TYPE] == A) | (_d2[TYPE] == Am) | (_d2[TYPE] == As)].pivot_table(index=[TIME],
    #                               values=[TARGET, NTargtAvg],
    #                               aggfunc=np.average).\
    #     plot().set_title("{}type {}, non taget".format(args.title, "A, Am, As"))
    # pp.draw()
    _d4 = _d2[(_d2[TYPE] == A) | (_d2[TYPE] == Am) | (_d2[TYPE] == As)].pivot_table(index=[TIME],
                                                                                   values=[
                                                                                       TARGET, NTargtAvg],
                                                                                   aggfunc=np.average)
    _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
    _p.bar(x=1500, height=10, width=10, color='black')
    _p.bar(x=2700, height=10, width=10, color='black')
    # _p.legend([TARGET, NTargtAvg, 'qend: 1500', 'target: 2700'])
    _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'])
    _p.set_title("{}type {}, non target".format(args.title, "A, Am, As"))
    

    for t in [C, E]:
        # _d3[_d3[TYPE] == t].pivot_table(index=[TIME],
        #                                 values=[TARGET, COMP, FILLAvg],
        #                                 aggfunc=np.average).\
        #     plot().set_title("{}type {},  Filler Avg".format(args.title, t))
        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TARGET, COMP, FILLAvg],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.bar(x=1500, height=10, width=10, color='black')
        _p.bar(x=2700, height=10, width=10, color='black')
        # _p.legend([TARGET, COMP, NTargtAvg, 'qend: 1500', 'target: 2700'])
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'])
        _p.set_title("{}type {}, filler average".format(args.title, t))


        pp.draw()

    # when we done drawing graphs
    pp.show()




if __name__ == "__main__":
    main()
