import FWCore.ParameterSet.Config as cms

source = cms.Source( "PoolSource"
, noEventSort        = cms.untracked.bool( True )
, duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
, fileNames          = cms.untracked.vstring()
, skipBadFiles       = cms.untracked.bool( True )
)
# maximum number of events
maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)
