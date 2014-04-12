import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaMuOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaMuSkimPath") #the selector name must be same as the path name in EXOMu_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOMu'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticamutest.root') # can be modified later in EXOMu_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaMuOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaHLTMuonFilter_*_*",
	  "keep *_exoticaRecoMuonFilter_*_*",
      )
    )
exoticaMuOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



