import FWCore.ParameterSet.Config as cms

process = cms.Process("FakeConditions")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
)        


process.load("RecoVertex.BeamSpotProducer.BeamSpotFakeParameters_cfi")

# if you want to use results from a text file
#process.BeamSpotFakeConditions.getDataFromFile = cms.bool(True)

process.beamspot = cms.EDAnalyzer("BeamSpotFromDB")


process.test = cms.Path( process.beamspot )


