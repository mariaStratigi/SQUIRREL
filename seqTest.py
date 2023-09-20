import numpy as np
import scipy.sparse as sp
import math
from SeqEnv import SQUIRRELEnvironment
#from tensorforce.environments import Environment
from tensorforce import Agent, Environment
from sklearn.preprocessing import MinMaxScaler



"""
#file: the name of the file containing the groups
#path: change as needed: the full path of the folder containing the file of the groups
#return: A list of Strings of the form: "groupId    groupSim" of all groups in the file
"""
def readGroups(file,path='C:/Users/jarim/Documents/work/rnn/data/'):
    groupIds = []
    with open(path+file) as f:
        for line in f:
            groupIds.append(line)
    return groupIds
"""
#groupId: A string of the form "groupId '\t' groupSim"
#The groupId is a concatenation of all the ids of the members
#return: Alist with the ids of the group members
"""
def getMembers(groupId):
    members = []
    (gr,sim) = groupId.split("\t")
    (m1,m2,m3,m4,m5) = gr.split("_")
    members.append(int(m1))
    members.append(int(m2))
    members.append(int(m3))
    members.append(int(m4))
    members.append(int(m5))
    #if the group size is larger than 5 uncommend as needed
    #members.append(int(m6))
    #members.append(int(m7))
    #members.append(int(m8))
    #members.append(int(m9))
    return members


def calcStats(sats):
    sum = 0.0
    max = -1000.0
    min = 1000.0

    for k in sats:
        sum = sum + k
        if k < min:
            min = k
        if k > max:
            max = k
    maxMin = max - min
    sum = sum / len(sats)
    dis= 1 - maxMin
    fScore = 2*((sum*dis)/(sum + dis))
    return (sum,maxMin,fScore)
"""
#Read the already calculated single recommendations for all group members and all rounds of recommendations
#returns: A dictionary with key: round of recommendations  value: a dictionary U
#dictionary U keys: user id value: dictionary R
#dictionary R keys: itemId value: score
"""
def getPredictions(file):
    groupInf = {}
    groupRec = {}
    rec = {}
    iter = 0
    count = 0
    with open(file+".txt") as f:
        for l in f:
            if "Iteration" in l:
                groupRec = {}
                count = 0
                continue
            count = count + 1
            rec = {}
            firstSplit = l.split("[")
            id = firstSplit[0]
            allRecs = firstSplit[1].replace("]","")
            allRecs = allRecs.replace("\n","")
            secondSplit = allRecs.split(",")
            for r in secondSplit:
                itm = r.split(":")
                rec[itm[0]] = itm[1]
            groupRec[id] = rec
            if count == 5:
                groupInf[iter] = groupRec
                iter = iter + 1
    return groupInf


def updateRecs(group, iter, recMovies):
    gr = group[iter]
    users = {}
    i = 0
    for u in gr:
        recs = gr[u]
        userRecs = {}
        for m in recs:
            if m in recMovies:
                continue
            else:
                userRecs[m] = recs[m]
        users[u] = userRecs
    return users

def computeIDCG(n):
    idcg = 0
    for i in range(n):
        idcg = idcg + math.log(2) / math.log(i+2)
    return idcg

def computeNDCG(user,group):
    dcg = 0.0
    idcg = computeIDCG(len(group))
    left_out = 0
    size = len(user)
    for i in range(size):
        item = user[i]
        if item in group:
            rank = i + 1 - left_out
            dcg = dcg + math.log(2) / math.log(rank + 1)
    return dcg/idcg

def getTopKItems(user,n):
    i = 0
    l = []
    for x in user.keys():
        if i == n:
            break
        l.append(x)
        i = i + 1
    return l

def computeDFH(user,group):
    i = 1
    flag = False
    for x in group:
        if x in user:
            flag = True
            break
        i = i + 1
    if flag:
        return 1 / math.log((i+1),2)
    else:
        return 0


#These are not needed and are not used. Used during initial experiments to track progression
files = []


#MovieLens
"""
files.append("ratingsStartingFull2003.csv")
files.append("sem_2004_1.csv")
files.append("sem_2004_2.csv")
files.append("sem_2005_1.csv")
files.append("sem_2005_2.csv")
files.append("sem_2006_1.csv")
files.append("sem_2006_2.csv")
files.append("sem_2007_1.csv")
files.append("sem_2007_2.csv")
files.append("sem_2008_1.csv")
files.append("sem_2008_2.csv")
files.append("sem_2009_1.csv")
files.append("sem_2009_2.csv")
files.append("sem_2010_1.csv")
files.append("sem_2010_2.csv")
files.append("sem_2011_1.csv")
"""

#GoodReads - Amazon

"""
files.append("start.csv")
files.append("chunks1.csv")
files.append("chunks2.csv")
files.append("chunks3.csv")
files.append("chunks4.csv")
files.append("chunks5.csv")
files.append("chunks6.csv")
files.append("chunks7.csv")
files.append("chunks8.csv")
files.append("chunks9.csv")
files.append("chunks10.csv")
files.append("chunks11.csv")
files.append("chunks12.csv")
files.append("chunks13.csv")
files.append("chunks14.csv")
"""

#Read all the groups in the file
groupsIds = readGroups('4_1GroupsTrain.txt')


print('Start training')
environment = Environment.create(
    environment=SQUIRRELEnvironment,
    max_episode_timesteps=15)
#agent = Agent.create(
#    agent='tensorforce', environment=environment, update=2,
#    optimizer=dict(optimizer='adam', learning_rate=3.2e-4),
#    objective='policy_gradient', reward_estimation=dict(horizon=1)
#)
agent = Agent.create(agent='ppo.json', environment=environment)

#****TRAINING******
j =1
#An array so as to calculate the number of times each action was selected
#If more actions are added this needs to be increased accordingly

actionsChoosenTrain = [0] * 6
for gr in groupsIds:

    grIn = gr.split("\t")
    print('Training Group ' + " (" + str(j) + ")\t" + grIn[0] )
    #groupInfo: the pre-calculated single recommendations for all group members and all rounds of recommendations
    #change the static link to the folder where you have stored the files in folders -> 3_2, 4_1 and 5Dif
    groupInfo = getPredictions("C:/Users/jarim/Documents/work/ephemeral/rnn/groupPredictions/allGroups/"+grIn[0].strip())

    #group: a list of all group member
    group = getMembers(gr)
    states = environment.reset()
    terminal = False
    internals = agent.initial_internals()
    i=0
    #recommentedMovies: the list saves all the items that have been recommended to the group throughout the recommendation rounds. 
    #By default items that have been recommended once to the group are not recommended again.
    recommentedMovies = []
    while not terminal:
        #groupRecs: the ratings of the group members without the items thar have been recommended
        groupRecs = updateRecs(groupInfo, i, recommentedMovies)

        actions = agent.act(states=states)
        #get the action selected
        actionsChoosenTrain[actions] = actionsChoosenTrain[actions] + 1
        #send the group members info to the environment
        environment.setUsers(groupRecs)
        #run the agent
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        #get the recommended items
        rec = environment.recs

        for k in rec:
            recommentedMovies.append(k)


        i = i + 1
    j = j + 1

print('training complete')


#*****TESTING******

### Initialize

states = environment.reset()

internals = agent.initial_internals()
terminal = False

### Run an episode

groupsIds = readGroups('4_1GroupsTest.txt')
satO = {}
maxMin = {}
fScore = {}
ndcg = {}
dfh = {}
j = 1
actionsChoosen  = [0] * 6
for groups in groupsIds:
    grIn = groups.split("\t")
    groupInfo = getPredictions("C:/Users/jarim/Documents/work/ephemeral/rnn/groupPredictions/allGroups/"+grIn[0].strip())
    print('Training Group ' + " (" + str(j) + ")\t" + grIn[0] )
    group = getMembers(groups)
    i = 0
    environment.reset()
    states = environment.en
    agent.reset()
    internals = agent.initial_internals()
    terminal = False
    recommentedMovies = []
    while not terminal:
        print('\tIteration: ' + str(i))
        groupRecs = updateRecs(groupInfo, i, recommentedMovies)

        environment.setUsers(groupRecs)
        actions, internals = agent.act(states=states, internals=internals, independent=True)
        states, terminal, reward = environment.execute(actions=actions)
        rec = environment.recs
        actionsChoosen[actions] = actionsChoosen[actions] + 1
        for k in rec:
            recommentedMovies.append(k)
        (sum,var,fScoreV) = calcStats(states)
        ndcgV = 0.0
        nn = 0
        for gl in groupRecs:
            glD = groupRecs[gl]
            ul = getTopKItems(glD,5) #here
            nd = computeNDCG(ul,rec)
            ndcgV = ndcgV + nd
            nn = nn + 1
        ndcgV = ndcgV / nn

        dfhV = 0.0
        nn = 0
        for gl in groupRecs:
            glD = groupRecs[gl]
            ul = glD.keys()
            nd = computeDFH(ul,rec)
            dfhV = dfhV + nd
            nn = nn + 1
        dfhV = dfhV / nn
        if j == 1:
            satOV = []
            satOV.append(sum)
            satO[str(i)] = satOV
            mm = []
            mm.append(var)
            maxMin[str(i)] = mm
            fs = []
            fs.append(fScoreV)
            fScore[str(i)] = fs
            nn = []
            nn.append(ndcgV)
            ndcg[str(i)] = nn
            dd = []
            dd.append(dfhV)
            dfh[str(i)] = dd
        else:
            satOV = satO[str(i)]
            satOV.append(sum)
            mm = maxMin[str(i)]
            mm.append(var)
            fs = fScore[str(i)]
            fs.append(fScoreV)
            nn = ndcg[str(i)]
            nn.append(ndcgV)
            dd = dfh[str(i)]
            dd.append(dfhV)
        i = i + 1
    j = j + 1
    fl = environment.flagEmpty
    if fl:
        print("Has empty values >>>>>>>>>>" + str(grIn[0]))
with open('AllGroups_Goodreads_FScore.txt', 'w') as f:
    f.write('Overall Satisfaction\n')

    for i in range(15):
        sum = 0
        j = 0
        tmp = satO[str(i)]
        for y in tmp:
            sum = sum + y
            j = j + 1
        sum = sum / j
        f.write(str(sum) + '\n')

    f.write('\nMaxMin\n')
    for i in range(15):
        sum = 0
        j = 0
        tmp = maxMin[str(i)]
        for y in tmp:
            sum = sum + y
            j = j + 1
        sum = sum / j
        f.write(str(sum) + '\n')

    f.write('\nNDCG\n')
    for i in range(15):
        sum = 0
        j = 0
        tmp = ndcg[str(i)]
        for y in tmp:
            sum = sum + y
            j = j + 1
        sum = sum / j
        f.write(str(sum) + '\n')

    f.write('\nDFH\n')
    for i in range(15):
        sum = 0
        j = 0
        tmp = dfh[str(i)]
        for y in tmp:
            sum = sum + y
            j = j + 1
        sum = sum / j
        f.write(str(sum) + '\n')

    f.write('\nFScore\n')
    for i in range(15):
        sum = 0
        j = 0
        tmp = fScore[str(i)]
        for y in tmp:
            sum = sum + y
            j = j + 1
        sum = sum / j
        f.write(str(sum) + '\n')

    f.close()
    print(">>>>>Actions TRAIN<<<<<")
    for act in actionsChoosenTrain:
        print(str(act))

    print("\n>>>>>Actions TEST<<<<<")
    for x in actionsChoosen:
        print(str(x))
