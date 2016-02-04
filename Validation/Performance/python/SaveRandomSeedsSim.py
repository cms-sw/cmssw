#G.Benelli Feb 7 2008
#This fragment is used to have the random generator seeds saved to test
#simulation reproducibility. Anothe fragment then allows to run on the
#root output of cmsDriver.py to test reproducibility.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']='SIMSavingSeeds'
    #Storing the random seeds
    #This would no more be necessary, randomEngineStateProducer is always in the configuration
    #But I prefer to delete it and call it rndStore:
    #del process.randomEngineStateProducer
    process.rndmStore=cms.EDProducer("RandomEngineStateProducer")
    #Adding the RandomEngine seeds to the content
    process.output.outputCommands.append("keep RandomEngineStates_*_*_*")
    process.rndmStore_step=cms.Path(process.rndmStore)
    #Modifying the schedule:
    #First delete the current one:
    del process.schedule[:]
    #Then add the wanted sequences
    process.schedule.append(process.simulation_step)
    process.schedule.append(process.rndmStore_step)
    process.schedule.append(process.out_step)
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Tweak Message logger to dump G4cout and G4cerr messages in G4msg.log
    process.MessageLogger.destinations=cms.untracked.vstring('warnings'
                                                             , 'errors'
                                                             , 'infos'
                                                             , 'debugs'
                                                             , 'cout'
                                                             , 'cerr'
                                                             , 'G4msg'
                                                             )
    process.MessageLogger.categories=cms.untracked.vstring('FwkJob'
                                                           ,'FwkReport'
                                                           ,'FwkSummary'
                                                           ,'Root_NoDictionary'
                                                           ,'G4cout'
                                                           ,'G4cerr'
                                                           )
    process.MessageLogger.cerr = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        )
    process.MessageLogger.G4msg =  cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True)
        ,threshold = cms.untracked.string('INFO')
        ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
        ,G4cout = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        ,G4cerr = cms.untracked.PSet(limit = cms.untracked.int32(-1))
        )
    return(process)
