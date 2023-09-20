from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np


###-----------------------------------------------------------------------------
### Environment definition
class SQUIRRELEnvironment(Environment):

    def __init__(self):
        
        self.en = []
        self.en.append(0.0)
        self.en.append(0.0)
        self.en.append(0.0)
        self.en.append(0.0)
        self.en.append(0.0)

        self.iterSat = []
        self.iterSat.append(0.0)
        self.iterSat.append(0.0)
        self.iterSat.append(0.0)
        self.iterSat.append(0.0)
        self.iterSat.append(0.0)
        self.flagEmpty = False
        #self.rec = []
        super().__init__()


    def states(self):
        return dict(type='float', shape=(5,), min_value=0.0, max_value=5.0)


    def actions(self):
        return dict(type='int', num_values=6)


    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional
    def close(self):
        super().close()


    def reset(self):
        """Reset state.
        """
        # state = np.random.random(size=(1,))
        self.timestep = 0
        #self.current_temp = np.random.random(size=(1,))
        self.en = []
        self.iterSat = []
        self.flagEmpty = False
        for i in range(5):
            self.en.append(0.0)
            self.iterSat.append(0.0)
        #return self.current_temp
        return self.en





    def getLeastSatUser(self):
        min = 1000.0
        ind = -1
        i = 0
        for x in self.en:
            if x < min:
                min = x
                ind = i
            i = i + 1
        return ind + 1



    def getAverage(self,item):
        sum = 0
        i = 0
        for u in self.users:
            i = i + 1
            rec = self.users[u]
            if item in rec.keys():
                sum = sum + float(rec[item])
        return sum / i



    def getLeastScore(self,item,ind):
        i = 0
        sc = 0
        for u in self.users:
            if i != ind:
                i = i + 1
                continue
            rec = self.users[u]
            if item in rec.keys():
                sc = float(rec[item])
            break
        return sc




    def getTopK(self,scores,k):
        sc = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse = True)}
        i = 0
        self.recs = []
        for it in sc.keys():
            self.recs.append(it)
            i = i + 1
            if i == k:
                break
        return self.recs



    def getTopKUser(self,scores, k):
        sc = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse = True)}
        i = 0
        sum = 0
        for x in sc.keys():
            i = i + 1
            sum = sum + float(sc[x])
            if i == k:
                break
        return sum



    def getRecRel(self, scores):
        sum = 0
        for x in self.recs:
            if x in scores.keys():
                sum = sum + float(scores[x])
        return sum

    def getRecRelPartial(self, scores, single):
        sum = 0
        for x in scores:
            if x in single.keys():
                sum = sum + float(single[x])
        return sum





    """
    Updates the satisfaction of the members based on the top 10 recommendations
    """
    def getSatisfactions(self, k):
        allS = []
        i=0
        for y in self.users:
            rec = self.users[y]
            arith = self.getRecRel(rec)
            para = self.getTopKUser(rec,k)
            sat = 0
            if para == 0:
                self.flagEmpty = True
                for x in rec.keys():
                    print(str(x) + "\t" + str(rec[x]))
            else:
                sat = arith / para
            satNew = self.en[i]+((sat- self.en[i])/self.timestep)
            self.en[i] = satNew
            allS.append(sat)
            i = i +1
        return allS



    def setUsers(self,users):
        self.users = users
        return
    """
    Returns a list of all unique items in all individual list of the Users
    """
    def setCandidates(self,usr):
        cand = []
        for u in usr:
            rec = usr[u]
            for r in rec:
                if r in cand:
                    continue
                cand.append(r)
        self.candidates = cand
        return


    """
    def computeReward(self,sats): #--->Fscore<--- between members _overall satisfaction_
        max = -100.0
        min = 100.0
        sum = 0
        for x in self.en:
            if x > max:
                max = x
            if x < min:
                min = x
            sum = sum + x
        sum = sum / len(self.en)
        maxMin = max - min
        maxMin = 1 - maxMin
        fScore = 2*((sum*maxMin)/(sum + maxMin))
        #print(str(fScore))
        return fScore
    """


    def computeReward(self, sats): #--->average<--- between members _iteration satisfaction_
        i = 0
        sum = 0
        for s in self.en:
            sum = sum + s
            i = i + 1
        reward = sum / i
        return reward


    """
    def computeReward(self, sats): #--->Variance<--- between members _iteration satisfaction_
        aa = np.array(list(sats))
        reward = np.var(aa)
        return 1-reward
    """
    #Calculates the alpha for the SDAA method
    def calcAlpha(self):
        max = -100.0
        min = 100.0
        for d in self.en:
            if d > max:
                max = d
            if d < min:
                min = d
        a = max - min
        return a
    """
    #Average aggregation method
    """
    def aggregateAvg(self):
        scores = {}
        i = 0
        for sc in self.candidates:
            avg = self.getAverage(sc)
            scores[sc] = avg
        return scores
    """
    #SDAA aggregation method
    """
    def aggregateSDAA(self):
        leastSat = self.getLeastSatUser()
        #scores = np.zeros(len(self.candidate))
        scores = {}
        i = 0
        a = self.calcAlpha()
        for sc in self.candidates:
            avg = self.getAverage(sc)
            lm = self.getLeastScore(sc,leastSat)

            gSc = (1-a)*avg
            gSc = gSc + a*lm
            scores[sc] = gSc
            i = i + 1
        return scores
    """
    SIAA aggregation method
    """
    def calcWeights(self):
        w = []
        max = -100.0
        for s in self.iterSat:
            if s > max:
                max = s
        for i in range(5):
            tmp = 0.8*(1-self.en[i]) + 0.2*(max-self.iterSat[i])
            w.append(tmp)
        return w
    """
    #SIAA aggregation method
    """
    def aggregateSIAA(self):
        scores = {}
        if self.timestep == 0:
            scores = self.aggregateAvg()
        else:
            weights = self.calcWeights()
            for sc in self.candidates:
                sum = 0
                i = 0
                for u in self.users:
                    rec = self.users[u]
                    if sc in rec.keys():
                        sum = sum + weights[i] * float(rec[sc])
                    i = i + 1
                sum = sum / i
                scores[sc] = sum
        return scores
    """
    Calculates partial satisfaction scores for the Avg+ aggregation method
    """
    def calcPartialSatScore(self,recGroup):
        allS = []
        i=0
        for y in self.users:
            rec = self.users[y]
            arith = self.getRecRelPartial(recGroup,rec)
            para = self.getTopKUser(rec,len(recGroup))
            sat = 0
            if para == 0:
                #print(str(y) + " size of recs: " + str(len(rec)) + " at iteration: " + str(self.timestep))
                self.flagEmpty = True
                sat = 0
            else:
                sat = arith / para
            allS.append(sat)
            #print(sat)
            i = i +1
        return allS

    def calcGroupDis(self,sats):
        max = -100.0
        min = 100
        for s in sats:
            if s > max:
                max = s
            if s < min:
                min = s
        dis = max - min
        return dis

    def aggregateAvgPlus(self):
        avgScores = self.aggregateAvg()
        cand = self.getTopK(avgScores,50)
        scores = {}
        while len(scores) < 5:  #here
            tmp = {}
            min = 100.0
            id = ""
            for sc in scores.keys():
                tmp[sc] = scores[sc]
            for sc in cand:
                if sc in scores.keys():
                    continue
                tmp[sc] = avgScores[sc]
                sats = self.calcPartialSatScore(tmp)
                dis = self.calcGroupDis(sats)
                if dis < min:
                    min = dis
                    id = sc
                del tmp[sc]
            scores[id] = min
        return scores


    def aggregatePareto(self):
        scores = {}

        while len(scores) < 5: #here
            tmp = {}
            max = -100.0
            id = ""
            for sc in scores.keys():
                tmp[sc] = scores[sc]
            for sc in self.candidates:
                if sc in scores.keys():
                    continue
                tmp[sc] = 0.0
                sats = self.calcPartialSatScore(tmp)
                sw = 0
                for s in sats:
                    sw = sw + s
                sw = sw / len(sats)
                f = np.var(sats)
                f = 1-f
                score = 0.8*sw + 0.2*f
                if score > max:
                    max = score
                    id = sc
                del tmp[sc]
            scores[id] = max
        return scores

    def aggregateSihem(self):
        scores = {}

        for sc in self.candidates:
            a = 0
            i = 0
            dis = 0
            for y in self.users:
                rec = self.users[y]
                o1 = 0
                if sc in rec.keys():
                    a = a + float(rec[sc])
                    o1 = float(rec[sc])
                j = 0
                for u in self.users:
                    if i == j:
                        continue
                    j = j + 1
                    rec2 = self.users[u]
                    o2 = 0
                    if sc in rec2.keys():
                        o2 = float(rec2[sc])
                    dis = dis + abs(o1-o2)
                i = i + 1
            dis = (2*dis)/(len(self.users)*(len(self.users)-1))
            a = a / len(self.users)
            score = 0.2*a + 0.8*dis
            scores[sc] = score
        return scores

    def execute(self,actions):
        ## Check the action is either 0 or 1 -- heater on or off.
        #assert actions == 0 or actions == 1

        ## Increment timestamp
        self.timestep += 1

        ## Update the current_temp
        #self.current_temp = self.response(actions)

        self.setCandidates(self.users)



        if actions == 0:
            m = self.aggregateAvg()
        elif actions == 1:
            m = self.aggregateSDAA()
        elif actions == 2:
            m = self.aggregateSIAA()
        elif actions == 3:
            m = self.aggregateAvgPlus()
        elif actions == 4:
            m = self.aggregatePareto()
        elif actions == 5:
            m = self.aggregateSihem()
        else:
            print("ERROR---ERROR")


        recs = self.getTopK(m,10)    #here
        satS = self.getSatisfactions(10) #here
        self.iterSat = satS
        ## Compute the reward
        reward = self.computeReward(satS)

        
        if self.timestep == 15:
            terminal = True
        else:
            terminal = False

        return self.en, terminal, reward
