import FWCore.ParameterSet.Config as cms

out = cms.OutputModule( "PoolOutputModule"
, fileName       = cms.untracked.string( 'patTuple.root' )
, SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring( 'p' ) )
, outputCommands = cms.untracked.vstring( 'drop *'
                                        , 'keep edmTriggerResults_*_*_*'
                                        , 'keep *_hltTriggerSummaryAOD_*_*'
                                        )
)

outpath = cms.EndPath(
  out
)
