import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaDiLeptonOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaDiMuonSkimPath","exoticaDiElectronSkimPath","exoticaEMuSkimPath") 
#the selector name must be same as the path name in EXODiLepton_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXODiLepton'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticadileptontest.root') # can be modified later in EXODiLepton_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaDiLeptonOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exoticaHLTDiMuonFilter_*_*",
	  "keep *_exoticaHLTDiElectronFilter_*_*",
	  "keep *_exoticaHLTElectronFilter_*_*",
	  "keep *_exoticaHLTMuonFilter_*_*",
	  "keep *_exoticaRecoDiMuonFilter_*_*",
	  "keep *_exoticaRecoDiElectronFilter_*_*",
	  "keep *_exoticaRecoMuonFilter_*_*",
	  "keep *_exoticaRecoElectronFilter_*_*",	  
      )
    )
exoticaDiLeptonOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



