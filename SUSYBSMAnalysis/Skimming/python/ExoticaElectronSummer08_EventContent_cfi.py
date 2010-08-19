import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaEOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaESkimPath") #the selector name to be same as the path name  in cfg.py
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ExoticaE'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('exoticaetest.root')
  )

#default output content AODSIMEventContent or AODEventContent
exoticaEOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specifi content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaERecoQalityCut_*_*"
      )
    )
exoticaEOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



