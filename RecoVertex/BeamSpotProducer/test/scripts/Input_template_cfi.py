import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(

'INPUTFILES'

	)
)

