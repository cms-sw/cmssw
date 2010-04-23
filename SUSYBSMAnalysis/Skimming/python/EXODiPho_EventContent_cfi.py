import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *

exoticaDiPhoOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaDiPhoSkimPath") #the selector name must be same as the path name in EXODiPho_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXODiPho'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticatest.root') # can be modified later in EXODiPho_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaDiPhoOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaHLTDiPhoFilter_*_*",
	  "keep *_exoticaRecoDiPhoFilter_*_*",
      )
    )
exoticaDiPhoOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



