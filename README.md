# SQUIRREL

seqTest: The main file of the code. Training and testing are implemented there.

SeqEnv: The class file for the environment of the model. Contains the declaration of the state and the implementation of the actions and reward functions

ppo.json: The policy that the rnn follows. Change it if it is needed.


--The folders contain the information we generated from the single recommender (here an Collaborative Filtering model). For each group we have all the items that the single
recommender has produced for all group members for 15 rounds of recommendations.

--In the Train and Test files we have the groups we evaluate. The first value is the group Id and the second is the group similarity. Only the id is used in the code.


4_1 folder: contains the prediction of the the 4_1 groups for the MovieLens dataset

4_1GroupsTrain: the groups used for training

4_1GroupsTest: the groups used for testing

5Dif folder: contains the prediction of the the 5 Different groups for the MovieLens dataset

5DifGroupsTrain: the groups used for training

5DifGroupsTest: the groups used for testing

3_2 folder: contains the prediction of the the 3_2 groups for the MovieLens dataset

3_2GroupsTrain: the groups used for training

3_2GroupsTest: the groups used for testing

4_1 folder: contains the prediction of the the 4_1 groups for the MovieLens dataset

4_1GroupsTrain: the groups used for training

4_1GroupsTest: the groups used for testing

allGroupsTest: contains groups from all group types (4_1, 3_2 and 5Dif) for testing
allGroupsTrain: contains groups from all group types (4_1, 3_2 and 5Dif) for training
