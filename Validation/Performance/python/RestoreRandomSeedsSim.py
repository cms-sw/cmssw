#G.Benelli Feb 7 2008
#This fragment is used to have the random generator seeds saved to test
#simulation reproducibility. Anothe fragment then allows to run on the
#root output of cmsDriver.py to test reproducibility.

import FWCore.ParameterSet.Config as cms
def customise(process):
    #Renaming the process
    process.__dict__['_Process__name']='SIMRestoringSeeds'
    #Skipping the first 3 events:
    process.PoolSource.skipEvents=cms.untracked.uint32(3)
    #Adding RandomNumberGeneratorService
    process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string('rndmStore')
    process.RandomNumberGeneratorService.VtxSmeared.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.g4SimHits.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.mix.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simSiPixelDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simSiStripDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simEcalUnsuppressedDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simHcalUnsuppressedDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simMuonCSCDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simMuonDTDigis.initialSeed = cms.untracked.uint32(1)
    process.RandomNumberGeneratorService.simMuonRPCDigis.initialSeed = cms.untracked.uint32(1)
    #This line is necessary to eliminate the "theSource" (i.e. source seed) in the python configuration!
    del process.RandomNumberGeneratorService.theSource
    #Adding the RandomEngine seeds to the content
    process.output.outputCommands.append("drop *_*_*_Sim")
    process.output.outputCommands.append("keep RandomEngineStates_*_*_*")
    process.g4SimHits_step=cms.Path(process.g4SimHits)
    #Modifying the schedule:
    #First delete the current one:
    del process.schedule[:]
    #Then add the wanted sequences
    process.schedule.append(process.g4SimHits_step)
    process.schedule.append(process.out_step)
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    return(process)
