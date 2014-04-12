import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaHTOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaHTSkimPath") #the selector name must be same as the path name in EXOHT_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOHT'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticaesinglejettest.root') # can be modified later in EXOHT_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaHTOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaHLTHTFilter_*_*",
      "keep *_exoticaRecoHTFilter_*_*",
      )
    )
exoticaHTOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



