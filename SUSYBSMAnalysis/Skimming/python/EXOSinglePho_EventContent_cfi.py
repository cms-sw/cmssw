import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

exoticaSinglePhoOutputModule = cms.OutputModule("PoolOutputModule",
                                          outputCommands = cms.untracked.vstring(),
#                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('exoticaSinglePhoHighetPath')),
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('exoticaRecoSinglePhoHighetPath')),
                                          dataset = cms.untracked.PSet(filterName = cms.untracked.string('EXOSinglePho'),
                                                                       dataTier = cms.untracked.string('EXOGroup')
    ),
                                          
                                          fileName = cms.untracked.string('exoticatest.root')
                                          )
#default output contentRECOSIMEventContent
exoticaSinglePhoOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

