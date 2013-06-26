import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

exoticaEleOutputModule = cms.OutputModule("PoolOutputModule",
                                          outputCommands = cms.untracked.vstring(),
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('exoticaEleLowetPath','exoticaEleMedetPath','exoticaEleHighetPath')),
                                          dataset = cms.untracked.PSet(filterName = cms.untracked.string('EXOEle'),
                                                                       dataTier = cms.untracked.string('EXOGroup')
    ),
                                          
                                          fileName = cms.untracked.string('exoticatest.root')
                                          )
#default output contentRECOSIMEventContent
exoticaEleOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

