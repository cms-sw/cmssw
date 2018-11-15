from __future__ import print_function
import FWCore.ParameterSet.Config as cms

class RandomRunSource (cms.Source):
    """The class is a Source whose run is chosen randomly.  This initializes identically to a cms.Source
    and after being initialized the run number distribution is set by calling 'setRunDistribution'.
    """
    def setRunDistribution(self,runsAndProbs):
        """Pass a list of tuple pairs, with the first item of the pair a run number
        and the second number of the pair a weight.  The class will normalize the
        weights so you do not have to.  The pairs will be used to randomly choose what Run
        should be assigned to the job.
        """
        self.__dict__['runsAndProbs']=runsAndProbs
    def insertInto(self, parameterSet, myname):
        from random import SystemRandom
        totalProb = 0.
        for r,p in self.__dict__['runsAndProbs']:
            totalProb+=p
        #this is the same random generator used to set the seeds for the RandomNumberGeneratorService
        random = SystemRandom()
        runProb = random.uniform(0,totalProb)
        print(runProb)
        sumProb = 0
        runNumber = 0
        for r,p in self.__dict__['runsAndProbs']:
            sumProb+=p
            if sumProb >= runProb:
                runNumber = r
                break
        if self.type_() == "PoolSource":
            self.setRunNumber = cms.untracked.uint32(runNumber)
        else:
            #sources that inherit from ConfigurableInputSource use 'firstRun'
            self.firstRun = cms.untracked.uint32(runNumber)
        super(RandomRunSource,self).insertInto(parameterSet,myname)
