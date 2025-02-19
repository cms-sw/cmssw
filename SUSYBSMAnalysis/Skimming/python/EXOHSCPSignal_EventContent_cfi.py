import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *

exoticaHSCPOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaHSCPDedxSkimPath","exoticaHSCPMuonSkimPath") #the selector name must be same as the path name in EXOHSCPSignal_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOHSCPSignal'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticahscptest.root') # can be modified later in EXOHSCP_cfg.py in  test directory. 
)


#default output contentRECOSIMEventContent
exoticaHSCPOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exotica*_*_*",      
      )
    )
exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)


