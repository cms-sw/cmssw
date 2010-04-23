import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaSingleJetOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaSingleJetSkimPath") #the selector name must be same as the path name in EXOSingleJet_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOSingleJet'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticasinglejettest.root') # can be modified later in EXOSingleJet_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaSingleJetOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaHLTSingleJetFilter_*_*",
	  "keep *_exoticaRecoSingleJetFilter_*_*",
      )
    )
exoticaSingleJetOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



