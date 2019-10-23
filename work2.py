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

## PART       = 'Participat'
## TRIAL      = 'Trial'
## TIME       = 'Time'
## TARGET     = 'Target'
## FILL1      = 'Filler 1'
## COMP       = 'Competitor'
## FILL2      = 'Filler 2'
## TYPE       = 'Type'
## VER        = 'Version'

_PART       = 'RECORDING_SESSION_LABEL'
PART        = 'Participant'
_TRIAL      = 'TRIAL_INDEX'
TRIAL       = 'Trial'
_TARGET     = 'AVERAGE_IA_1_SAMPLE_COUNT'
TARGET      = 'Target'
_FILL1      = 'AVERAGE_IA_2_SAMPLE_COUNT'
FILL1       = 'Filler1'
_COMP       = 'AVERAGE_IA_3_SAMPLE_COUNT'
COMP        = 'Competitor'
_FILL2      = 'AVERAGE_IA_4_SAMPLE_COUNT'
FILL2       = 'Filler2'
_TIME       = 'BIN_START_TIME'
TIME        = 'Time'
TYPE        = 'type'
VERSION     = 'version'
TARGET_WORD = 'target_word'

# calculated special
FILLAvg    = 'Filler Average'
NTargtAvg  = 'Non Target Average'
TCAvg      = 'Target Competitor Average'
TmC        = 'Target minus Competitor'
TmNT       = 'Target minus Non Target'
TCmF       = 'Target Competitor minus Filler'

KEEP_COLUMNS = [TIME, PART, TRIAL, TYPE, VERSION, TARGET, COMP, FILL1, FILL2, TARGET_WORD,
                FILLAvg, NTargtAvg, TCAvg, TmC, TmNT, TCmF]

TARGET_COLOR  = 'red'
COMP_COLOR    = 'blue'
NTargetCOLOR  = 'green'
color_dict={COMP: COMP_COLOR, TARGET: TARGET_COLOR, NTargtAvg: NTargetCOLOR, FILLAvg: NTargetCOLOR, TCAvg: TARGET_COLOR}

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

TITLE_FONT_SIZE=20
LEGEND_FONT_SIZE=12

# output_name=args.input.replace('.xlsx', '_result.xlsx')

#pdb.set_trace()

def get_data_raw(input) -> pd.DataFrame:
    workSet = None

    csv_input =  input.split(".")[0] + ".csv"
    if os.path.isfile(csv_input):
        workset = pd.read_csv(csv_input)
    
    if workSet == None:
        try:
            data = pd.read_excel(input, sheet_name=None) # read all sheets
            if 'Sheet1' not in data:
                print("Sheet1 not found in excel sheets, geting the first sheet\n")
                workSet = data[list(data.keys())[0]].copy(deep=True)
            else:
                workSet = data['Sheet1'].copy(deep=True)
        except Exception as e:
            print("failed to read {} as excel file, trying as csv with tab seperator\n".format(input))
            workSet = pd.read_csv(input, sep="\t")

    workSet.rename(columns={
        _PART:   PART,
        _TRIAL:  TRIAL,
        _TARGET: TARGET,
        _COMP:   COMP,
        _FILL1:  FILL1,
        _FILL2:  FILL2,
        _TIME:   TIME
    }, inplace=True)
    workSet.to_csv(csv_input, index=False)
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

    csv_input = input.split(".")[0] + ".csv"

    # input = "060919/old_message_report.xlsx"
    # input = args.input

    if os.path.isfile(csv_input):
        _data = pd.read_csv(csv_input)
    else:
        _d = pd.read_excel(input, sheet_name=None)
        _data = _d['all_trial_messages']
        _data.to_csv(csv_input, index=False)
    
    _data.rename(columns={
        _PART:   PART,
        _TRIAL:  TRIAL
    }, inplace=True)

    # filter messages which are not TOUCH_TARGET nor stimuli ...
    _data2 = _data.loc[(_data['CURRENT_MSG_TEXT'] == 'TOUCH_TARGET') | (
        _data['CURRENT_MSG_TEXT'].str.contains('Stimuli'))]
    
    _data2[_data2['CURRENT_MSG_TEXT'].str.contains('Stimu')].pivot_table(
        index=[PART, TRIAL], values='CURRENT_MSG_TIME', aggfunc=min)
    
    # find which trials are missing entries of stimuli touch report bundle
    _data3 = _data2.pivot_table(index=[PART], values=[
                                'CURRENT_MSG_TEXT'], columns=[TRIAL], aggfunc=[len])
    
    # rename value from 'Stimuli: <NAME>' to just 'Stimuli'
    _data4 = _data2.replace({'Stimuli:.*': 'Stimuli'}, regex=True)
    
    # pivot table to have columns for TOUCH_TARGET and Stimuli, have min value of CURRENT_MSG_TIME as the new value
    _data5 = _data4.pivot_table(index=[PART, TRIAL], columns=[
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
    # _data7 = _data6.rename_axis([PART, TRIAL])

    return _data6

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
    p.set_ylim(-10, 100)
    p.set_title("Part: {}, Trial: {}, Type: {}".format(
        _p, _t, data2.loc[_p].loc[_t].index[0]), fontsize=TITLE_FONT_SIZE)
    p.legend(['Target', 'cutoff {}'.format(
        data2.loc[_p].loc[_t].TOUCH_FIXED.values[0]), 'qend: 1500', 'target: 2700'], loc='upper left',
        fontsize=LEGEND_FONT_SIZE)


def filter_no_gaze_to_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    filter the data according a specific set of conditions
    and return the data after filter was applied
    """
    d = data
    condStart = (d[TIME] >= 200)
    condStop1 = (d[TIME] <= 3500)
    condStop2 = (d[TIME] <= 4200)
    condTypeCombined = ((d[TYPE] == F) | (d[TYPE] == AF))
    condNormal = condStart & condStop1 & ~condTypeCombined
    condCombined = condStart & condStop2 & condTypeCombined

    nd = d[condNormal]
    cd = d[condCombined]

    for _d in [nd, cd] :

        # find participants trials that are missing gaze to target
        # reduced normal data
        r_d = _d.pivot_table(index=[PART, TRIAL, TYPE], values=[TARGET], aggfunc=np.max)

        # missing gazes reduced normal data
        mr_d = r_d[r_d[TARGET] == 0]

        # normal work set summary of trials with missing gazes per participant
        # summary of missing gazes trials per participant on reduced normal data
        smr_d = mr_d.pivot_table(
            index=[PART],
            values=[TARGET],
            columns=[TYPE],
            aggfunc='count',
            # aggfunc=lambda x:  len([1 for z in x if z == 0]),  # naive count like
            dropna=True,
            fill_value=0)

        # total of summary of missing gazes on reduced normal data - total of trials per type which counted as missing gazes
        tsmr_d = smr_d[0:0].append(smr_d.sum().rename('Total'))

        # summary 2 : show breakdown of trials for participants
        _smr_d = smr_d.assign(**{'total per participant': smr_d.sum(axis=1)})
        s2mr_d = _smr_d.append(_smr_d.sum().rename('Total'))

        # get the list of participants and trials which we wish to ignore
        _t = r_d.pivot_table(index=[PART, TRIAL], values=[TARGET], aggfunc=np.max)

        # subtotals of trials with trial number per participant
        s3mr_d = pd.concat([
            d.append(d.count().rename((k, 'subtotal')))
            for k, d in _t[_t[TARGET] == 0].groupby(level=0)
        ])

        # ignored missing gaze trials of normal data
        im_d = _t[_t[TARGET] == 0].reset_index().drop(TARGET, axis=1)

        # omit
        # imnd_tuple_list = list(imnd.apply(tuple, axis=1))

        # filter the data of the ignore participants and trials
        # merge the tables and add indicate column where it contain:
        #  'left_only' for rows that exists on left only table
        #  'both' for rows that exists on both tables
        _f_d = pd.merge(d, im_d, on=[PART, TRIAL], how='left', indicator='Exist')
        # we want only 'left_only' and drop the indicate column
        f_d = _f_d[_f_d.Exist == 'left_only'].drop('Exist', axis=1)

        d = f_d

    return d

def process_data(data, touch_data, export) -> pd.DataFrame:
    
    data_with_touch = pd.merge(data.set_index([PART, TRIAL]),
                               touch_data,
                               on=[PART, TRIAL])

    # if we want to plot target for specific participant and trial
    # plot_part_trial(data_with_touch.set_index(PART, TRIAL, TYPE), 411, 5)

    _d = data_with_touch

    # pad Target column from touch point
    _d.loc[_d[TIME] >= _d['TOUCH_FIXED'], TARGET] = 10

    # normalized to 100
    for c in [TARGET, COMP, FILL1, FILL2]:
        _d[c] = _d[c] * 10

    # add custom calculated columns
    _d = _d.assign(**{
        NTargtAvg: (_d[COMP] + _d[FILL1] +_d[FILL2]) / 3,
        FILLAvg:   (_d[FILL1] + _d[FILL2]) / 2,
        TCAvg:     (_d[TARGET] + _d[COMP]) / 2
    })
    _d = _d.assign(**{
        TmC:       (_d[TARGET] - _d[COMP]),
        TmNT:      (_d[TARGET] - _d[NTargtAvg]),
        TCmF:      (_d[TCAvg] - _d[FILLAvg])
    })

    _d = filter_no_gaze_to_target(_d)

    for col in _d.columns:
        if col not in KEEP_COLUMNS:
            del _d[col]

    if export is not None:
        if export.split(".")[1] == "xlsx":
            export_csv = export.split(".")[0] + ".csv"
            #_d.reset_index().to_excel(export)
            _d.reset_index().to_csv(export_csv, index=False)
        elif export.split(".")[1] == "csv":   # TODO: make this condition nicer
            _d.reset_index().to_csv(export_csv, index=False)
        else:
            print("not exported to {}".format(export))

    return _d

def plot_graphs(data, title_prefix, outfolder):
    # chop time for  A(A, Am, As), B, C, D, E  (from 200 to 3500)
    _d2 = data[(data[TIME] >= 200) & (data[TIME] <= 3500)]
    # chop time for  AF, F  (from 200)
    _d3 = data[(data[TIME] >= 200)]

    trial_types = data[TYPE].unique()

    # look at B and C on the same grpah - the effect of with and without competition on 
    def B_and_C_together():
        # green - B
        # purple - C

        _dtb = _d2[_d2[TYPE] == B].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        _dtc = _d2[_d2[TYPE] == C].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        _p = _dtb.plot(color='green')
        _p.plot(_dtc, color='purple')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(['w/o competition', 'with competition', 'qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, target".format(title_prefix, 'B and C'), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, 'B C'))
        pp.draw()

    B_and_C_together()

    for t in [B, D]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TARGET, NTargtAvg],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, non target".format(title_prefix, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, t))
        pp.draw()

    def grouping_type_D():
        t = D
        d = _d2[_d2[TYPE] == t]
        # participant grouping - 2 groups - comparison in type D
        group1 = [1111, 1101, 2208, 2207, 6629, 6630, 6623] # worst hearing
        group2 = [] # all the rest

        if set(group1).intersection(set(_d2[PART].unique())) != set(group1):
            # not the right data
            return

        dg1 = d[d[PART].isin(group1)].pivot_table(index=[TIME],
                                                  values=[TARGET, NTargtAvg],
                                                  aggfunc=np.average)

        dg2 = d[~d[PART].isin(group1)].pivot_table(index=[TIME],
                                                   values=[TARGET, NTargtAvg],
                                                   aggfunc=np.average)

        for _d in [(dg1, 'worst hearing'), (dg2, 'better hearing')]:
            _p = _d[0].plot(color=[color_dict.get(x, "#333333") for x in _d[0].columns])
            _p.axvline(1500, color='black', linestyle=':')
            _p.axvline(2700, color='black', linestyle='--')
            _p.legend(list(_d[0].columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
            _p.set_ylim(-10, 100)
            name = title_prefix + _d[1]
            _p.set_title("{} type {}, non target".format(name, t), fontsize=TITLE_FONT_SIZE)
            _p.figure.set_size_inches(15, 9)
            if outfolder is not None:
                _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
            pp.draw()

    grouping_type_D()

    for t in [A, Am, As, 'A(all)']:
        # draw A type plot
        if t == 'A(all)':
            Aall = set([A, Am, As])
            if Aall.intersection(set(trial_types)) != Aall:
                print("trials of type {} not exists in data".format(Aall))
                continue
            cond = (_d2[TYPE] == A) | (_d2[TYPE] == Am) | (_d2[TYPE] == As)
        else:
            if t not in trial_types:
                print("trial of type {} not exists in data".format(t))
                continue
            cond = (_d2[TYPE] == t)

        _d4 = _d2[cond].pivot_table(index=[TIME],
                                    values=[TARGET, NTargtAvg],
                                    aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, non target".format(title_prefix, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, t))
        pp.draw()
    

    for t in [C, E]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TARGET, COMP, FILLAvg],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, filler average".format(title_prefix, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, t))
        pp.draw()

    def grouping_type_C():
        t = C
        d = _d2[_d2[TYPE] == t]
        # trial grouping - 2 groups - comparison in type C
        group1 = [85, 86, 94, 101] # last trials 
        group2 = [] # all the rest

        dg1 = d[d[TRIAL].isin(group1)].pivot_table(index=[TIME],
                                                     values=[
                                                         TARGET, COMP, FILLAvg],
                                                     aggfunc=np.average)

        dg2 = d[~d[TRIAL].isin(group1)].pivot_table(index=[TIME],
                                                      values=[
                                                          TARGET, COMP, FILLAvg],
                                                      aggfunc=np.average)

        for _d in [(dg1, 'last trials group'), (dg2, 'first trials group')]:
            _p = _d[0].plot(color=[color_dict.get(x, "#333333") for x in _d[0].columns])
            _p.axvline(1500, color='black', linestyle=':')
            _p.axvline(2700, color='black', linestyle='--')
            _p.legend(list(_d[0].columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
            _p.set_ylim(-10, 100)
            name = title_prefix + _d[1]
            _p.set_title("{} type {}, non target".format(name, t), fontsize=TITLE_FONT_SIZE)
            _p.figure.set_size_inches(15, 9)
            if outfolder is not None:
                _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
            pp.draw()

    grouping_type_C()

    for t in [C]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TARGET],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target only"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in [C]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d2[_d2[TYPE] == t].pivot_table(index=[TIME],
                                        values=[TCAvg, FILLAvg],
                                        aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['qend: 1500', 'target: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target competitor avg vs filler avg"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in [F]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d3[_d3[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TARGET, COMP, NTargtAvg],
            aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(4200, color='black', linestyle='--')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(3000, color='black', linestyle='-.')
        _p.legend(list(_d4.columns) + ['1qend: 1500', '2qend: 3000', 'target: 4200'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, non target".format(title_prefix, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, t))
        pp.draw()

    for t in [AF]:
        if t not in trial_types:
            print("trial of type {} not exists in data".format(t))
            continue

        _d4 = _d3[_d3[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TARGET, NTargtAvg],
            aggfunc=np.average)
        _p = _d4.plot(color=[color_dict.get(x, "#333333") for x in _d4.columns])
        _p.axvline(4200, color='black', linestyle='--')
        _p.legend(list(_d4.columns) + ['target: 4200'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, non target".format(title_prefix, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, t))
        pp.draw()

def plot_comparison_graphs(workset, outfolder, title_prefix):

    do = workset['old']
    # do = workset['old'].assign(TmNT=lambda x: x[TARGET] - x[NTargtAvg])
    # do = do.assign(TmC=lambda x: x[TARGET] - x[COMP])
    dy = workset['young']
    # dy = workset['young'].assign(TmNT=lambda x: x[TARGET] - x[NTargtAvg])
    # dy = dy.assign(TmC=lambda x: x[TARGET] - x[COMP])

    # chop time for  A(A, Am, As), B, C, D, E  (from 200 to 3500)
    _do2 = do[(do[TIME] >= 200) & (do[TIME] <= 3500)]
    _dy2 = dy[(dy[TIME] >= 200) & (dy[TIME] <= 3500)]
    # chop time for  AF, F  (from 200)
    _do3 = do[(do[TIME] >= 200)]
    _dy3 = dy[(dy[TIME] >= 200)]

    def ovsy_B_and_C_together():
        _dotb = _do2[_do2[TYPE] == B].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        _dotc = _do2[_do2[TYPE] == C].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        _dytb = _dy2[_dy2[TYPE] == B].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        _dytc = _dy2[_dy2[TYPE] == C].pivot_table(index=[TIME],
                                               values=[TARGET],
                                               aggfunc=np.average)
        all = pd.merge(_dotb, _dotc, on=[TIME]).merge(_dytb, on=[TIME]).merge(_dytc, on=[TIME])
        all.columns = ['old no comp', 'old comp', 'young no comp', 'young comp']
        _p = all.plot(style=['-', ':' ,'-', ':'], color=['magenta', 'magenta', 'cyan', 'cyan'])
        _p.axvline(1500, color='black', linestyle=':', label='qend: 1500')
        _p.axvline(2700, color='black', linestyle='--', label='target onset: 2700')
        _p.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        _p.set_title("{} type {}, target".format(title_prefix, 'B and C'), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(title_prefix, 'B C'))
        pp.draw()

    ovsy_B_and_C_together()

    for t in [B, C, D]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do2[_do2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmNT],
            aggfunc=np.average
        )
        _y = _dy2[_dy2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmNT],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + "target minus non target"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in ['A(all)']:
        Aall = set([A, Am, As])
        if Aall.intersection(set(do[TYPE].unique())) != Aall or Aall.intersection(set(dy[TYPE].unique())) != Aall:
            print("trials of type {} not exists in data".format(Aall))
            continue
        ocond = (_do2[TYPE] == A) | (_do2[TYPE] == Am) | (_do2[TYPE] == As)
        ycond = (_dy2[TYPE] == A) | (_dy2[TYPE] == Am) | (_dy2[TYPE] == As)

        _o = _do2[ocond].pivot_table(
            index=[TIME],
            values=[TmNT],
            aggfunc=np.average
        )
        _y = _dy2[ycond].pivot_table(
            index=[TIME],
            values=[TmNT],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + "target minus non target"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()
    
    for t in [B, C, E]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do2[_do2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmC],
            aggfunc=np.average
        )
        _y = _dy2[_dy2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmC],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target minus competitor"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in [B, C]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do2[_do2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TARGET],
            aggfunc=np.average
        )
        _y = _dy2[_dy2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TARGET],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target only"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in [B, C]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do2[_do2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TCAvg],
            aggfunc=np.average
        )
        _y = _dy2[_dy2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TCAvg],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target competitor avg"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

    for t in [B, C, D, E]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do2[_do2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TCmF],
            aggfunc=np.average
        )
        _y = _dy2[_dy2[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TCmF],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(2700, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['qend: 1500', 'target onset: 2700'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + " target competitor minus filler"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()


    for t in [F]:
        if t not in do[TYPE].unique() or t not in dy[TYPE].unique():
            print("trial of type {} not exists in data (young or old)".format(t))
            continue

        _o = _do3[_do3[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmC],
            aggfunc=np.average
        )
        _y = _dy3[_dy3[TYPE] == t].pivot_table(
            index=[TIME],
            values=[TmC],
            aggfunc=np.average
        )
        _p = _o.plot(color='magenta')
        _p.plot(_y, color='cyan')
        _p.axvline(1500, color='black', linestyle=':')
        _p.axvline(3000, color='black', linestyle=':')
        _p.axvline(4200, color='black', linestyle='-.')
        _p.legend(['old', 'young'] + ['1qend: 1500', '2qend: 3000', 'target onset: 4200'], loc='upper left', fontsize=LEGEND_FONT_SIZE)
        _p.set_ylim(-10, 100)
        name = title_prefix + "target minus competitor"
        _p.set_title("{} type {}".format(name, t), fontsize=TITLE_FONT_SIZE)
        _p.figure.set_size_inches(15, 9)
        if outfolder is not None:
            _p.figure.savefig(outfolder + "/{} {}.png".format(name, t))
        pp.draw()

def interactive(workset):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
    parser.add_argument("-iold", dest="olddata", type=str, help="old data excel file")
    parser.add_argument("-ioldtouch", dest="oldtouch", type=str, help="old touch data excel file")
    parser.add_argument("-oold", dest="outold", type=str, help="output old data excel file")
    parser.add_argument("-iyoung", dest="youngdata", type=str, help="young data excel file")
    parser.add_argument("-iyoungtouch", dest="youngtouch", type=str, help="young touch data excel file")
    parser.add_argument("-oyoung", dest="outyoung", type=str, help="output young data excel file")
    parser.add_argument("-t", dest="title", type=str, help="graph title prefix")
    parser.add_argument("-of", dest="outfolder", type=str, help="folder for saving graphs")
    parser.add_argument("-k", dest="keep", default=False, action='store_true', help="keep the grpahs onscreen")
    parser.add_argument("--interactive", dest="interactive", default=False, action='store_true', help="experimental interactive CLI")
    parser.add_argument("-ioldp", dest="olddatap", type=str, help="old data excel file preprocessed")
    parser.add_argument("-iyoungp", dest="youngdatap", type=str, help="young data excel file preprocessed")

    args = parser.parse_args()

    if args.outfolder is not None and not os.path.isdir(args.outfolder):
        os.makedirs(os.path.abspath(args.outfolder))

    workset=dict()
    if args.olddatap is not None:
        workset['old'] = pd.read_csv(args.olddatap)
    elif args.olddata is not None and args.oldtouch is not None:
        data = get_data_raw(args.olddata)
        touch_data = get_message_report_ofs(args.oldtouch)

        workset['old'] = process_data(data, touch_data, args.outold)
    
    if args.youngdatap is not None:
        workset['young'] = pd.read_csv(args.youngdatap)
    if args.youngdata is not None and args.youngtouch is not None:
        data = get_data_raw(args.youngdata)
        touch_data = get_message_report_ofs(args.youngtouch)

        data = data[~data[PART].isin([206, 605, 406, 409, 201, 213, 404])]
        workset['young'] = process_data(data, touch_data, args.outyoung)

    title = args.title if args.title is not None else ""
    if 'old' in workset:
        plot_graphs(workset['old'], title + " old ", args.outfolder)
    if 'young' in workset:
        plot_graphs(workset['young'], title + " young ", args.outfolder)


    if 'old' in workset and 'young' in workset:
        plot_comparison_graphs(workset, args.outfolder, title + " ovsy ")


    if args.keep is True:
        # when we done drawing graphs
        pp.show()
    elif args.interactive:
        interactive(workset)


if __name__ == "__main__":
    main()
