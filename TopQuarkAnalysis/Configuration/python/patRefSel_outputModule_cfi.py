import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.Configuration.patRefSel_eventContent_cff import common_eventContent

out = cms.OutputModule(
  "PoolOutputModule"
, fileName       = cms.untracked.string( 'test.root' )
, SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring() )
, outputCommands = cms.untracked.vstring( 'drop *'
                                        , *common_eventContent
                                        )
, dropMetaData   = cms.untracked.string( 'ALL' )
)
