
import csv
from itertools import groupby
import sys
import time

#dst = str(int(time.time())) + '_rawdata.csv'

if len(sys.argv) == 1:
    print("no enough arguments")
    sys.exit(1)

src = sys.argv[1]
prefix = src.split('.')[0]

padding = False
now = str(int(time.time()))
if len(sys.argv) > 2:
    dst = sys.argv[2]
    # TODO: thats ugly, make it nicer .. 
    if len(sys.argv) > 3 and sys.argv[3] == "pad":
        dst =  "padding_" + dst
        padding = True
else:
    dst =  'rawdata_'+ now +'.csv'


# create the data structure for the csv data
z=dict()
with open(src, 'r') as file:
    data = csv.reader(file)
    next(data)
    for r in data:
        r_0_part = int(r[0])                        # extract participant
        r_1_trial = int(r[1])                       # extract trial
        r_2_7 = [int(x) for x in r[2:7]]            # extract time
        r_2_7.append(r[7])                          # extract target , filler1, competitor, filler2, type                      
        if r_0_part not in z.keys():                
            z[r_0_part] = dict()                    # create dict value for key participant if not exists
        if r_1_trial not in z[r_0_part].keys():
            z[r_0_part][r_1_trial] = list()         # create list value for key trial for participant if trial not exist
        z[r_0_part][r_1_trial].append(r_2_7)        # populate trial data

# create a working copy
a=z.copy()

# trim edges of not intersting periods
# EARLY_TRIM - no eye movments yet
# TARGET_WORD_TIME - when the target time start to heard
# LATE_TRIM - well after the target (enough time after target word was heard)

EARLY_TRIM = 200
TARGET_WORD_TIME = 2700
LATE_TRIM_1 = 3500 # for normal sentences
LATE_TRIM_2 = 4500 # for combined sentenced (trial type f or af)

for k in a.keys():
    for kk in a[k].keys():
        a[k][kk] = [ x for x in a[k][kk] if (x[0] >= EARLY_TRIM and x[0] <= LATE_TRIM_1) ]

# check that trials have gaze to the target (avoid periferal looking or error)
# test that 1 trial for specific participant have at least 1 value of 10 in target

# if not any([x[1]>9 for x in a[max(a.keys())][1]]):
#     raise Exception("participant {} trial 1 do not have 10 value for target".format(x))

# verify all trials on all participants have at least 1 element value of target that have the value of 10
c=map(lambda aaa: map(lambda aa:  any([x[1]>9 for x in a[aaa][aa]]), a[aaa].keys()), a.keys())

if not all(c):
    raise Exception("some participant in some trial do not have 10 value for target")


# TODO: do we still need this section  ?
#
# #https://stackoverflow.com/questions/38161606/find-the-start-position-of-the-longest-sequence-of-1s
# # find index of the longest sequnce of largest value in the list
# foo = lambda L: max(((lambda y: (y[0][0], len(y)))(list(g)) for k, g in groupby(enumerate(L), lambda x: x[1]) if k), key=lambda z: z[1])[0]

# # b list of the positions of the start of the longest sequence 
# # c list of rouple of the  participant, trial where there is no indication of value other than 0
# c=list();b=list();
# for k in a.keys():
#     for kk in a[k].keys():
#         #print("checking:", k, kk)
#         L=[x[1] for x in a[k][kk]]
#         if max(L) > 0:
#             b.append(foo(L))
#         else:
#             #print("no max for:", k, kk)
#             c.append((k,kk))


# b list of tuples (participants, trials) which have no gaze at target
# c list of tuples (participants, trials, time index of last value of 10 in target)
c=list();b=list();
for k in a.keys():
    for kk in a[k].keys():
        #print("checking:", k, kk)
        L=[a[k][kk].index(x) for x in a[k][kk] if x[1] == 10]
        if len(L) > 0:
            b.append((k, kk, max(L)))
        else:
            #print("no max for:", k, kk)
            c.append((k,kk))

# pad in data for each trial from the time index of the last value of 10 in target 
# value till the end of the list (all the rest of the samples), 
# also 0 all non target (competitor and fillers)
# TODO: changet to paramater for the script
if padding:
    for e in b:
        for i in range(e[2], 250, 1):
            if i < len(a[e[0]][e[1]]):
                a[e[0]][e[1]][i][1]=10
                a[e[0]][e[1]][i][2]=0
                a[e[0]][e[1]][i][3]=0
                a[e[0]][e[1]][i][4]=0

# normalize to 100
for k in a.keys():
    for kk in a[k].keys():
        for e in a[k][kk]:
            e[1] *= 10
            e[2] *= 10
            e[3] *= 10
            e[4] *= 10


missing_gaze_at_target_dst = prefix + "_missing_gaze_at_target_" + dst
print("writing reuslt of trials which dont have value > 0 in target to: %s" % missing_gaze_at_target_dst)
with open(missing_gaze_at_target_dst, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["participant", "trial"])
    for e in c:
        writer.writerow(e)

dst1 = prefix + "_" + dst
print("writing result to: %s" % dst1)

with open(dst1, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["participant", "trial", "time", "target", "filler 1", "competitor", "filler 2", "type"])
    for k in a.keys():
        for kk in a[k].keys():
            if (k, kk) in c:
                continue
            for e in a[k][kk]:
                writer.writerow([k, kk] + e)

dst2 = prefix + "_filler_average_" + dst
print("writing filler average result to: %s" % dst2)
with open(dst2, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["participant", "trial", "time", "target", "filler average", "competitor", "type"])
    for k in a.keys():
        for kk in a[k].keys():
            if (k, kk) in c:
                continue
            for e in a[k][kk]:
                e2 = [e[0], e[1], (e[2] + e[4])/2, e[3], e[5]]
                writer.writerow([k, kk] + e2)

dst3 = prefix + "_non_target_average_" + dst
# todo work on below
print("writing non target average result to: %s" % dst3)
with open(dst3, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["participant", "trial", "time", "target", "non target average", "type"])
    for k in a.keys():
        for kk in a[k].keys():
            if (k, kk) in c:
                continue
            for e in a[k][kk]:
                e2 = [e[0], e[1], (e[2] + e[3] + e[4])/3, e[5]]
                writer.writerow([k, kk] + e2)


dst4 = prefix + "_target_minus_competitor_" + dst
print("writing target minus competitor result to: %s" % dst3)
with open(dst4, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["participant", "trial", "time", "target_minus_competitor", "type"])
    for k in a.keys():
        for kk in a[k].keys():
            if (k, kk) in c:
                continue
            for e in a[k][kk]:
                e2 = [e[0], e[1] - e[3], e[5]]
                writer.writerow([k, kk] + e2)