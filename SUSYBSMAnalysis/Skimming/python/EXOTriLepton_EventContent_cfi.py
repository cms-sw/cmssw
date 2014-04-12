import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *


exoticaTriLeptonOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaTriMuonSkimPath","exoticaTriElectronSkimPath","exotica1E2MuSkimPath","exotica2E1MuSkimPath") 
#the selector name must be same as the path name in EXOTriLepton_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOTriLepton'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticatrileptontest.root') # can be modified later in EXOTriLepton_cfg.py in  test directory. 
  )


#default output contentRECOSIMEventContent
exoticaTriLeptonOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "keep *_exotica*_*_*",
      )
    )
exoticaTriLeptonOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)



