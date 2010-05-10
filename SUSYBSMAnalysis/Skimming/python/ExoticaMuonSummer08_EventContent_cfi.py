import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaMuOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaMuSkimPath") #the selector name to be same as the path name  in cfg.py
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ExoticaMu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('exoticamutest.root')
  )


#default output content RecoEventContent
exoticaMuOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specifi content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaMuRecoQalityCut_*_*"
      )
    )
exoticaMuOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



