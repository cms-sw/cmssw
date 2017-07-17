import FWCore.ParameterSet.Config as cms

source = cms.Source(
  "PoolSource"
, fileNames     = cms.untracked.vstring()
, skipBadFiles  = cms.untracked.bool( True )
)
# maximum number of events
maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)
