import FWCore.ParameterSet.Config as cms

out = cms.OutputModule(
  "PoolOutputModule"
, fileName       = cms.untracked.string( 'test.root' )
, SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring() )
, outputCommands = cms.untracked.vstring( 'drop *'
                                        , 'keep edmTriggerResults_*_*_*'
                                        , 'keep *_hltTriggerSummaryAOD_*_*'
                                        )
, dropMetaData   = cms.untracked.string( 'ALL' )
)
