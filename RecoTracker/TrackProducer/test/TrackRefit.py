import FWCore.ParameterSet.Config as cms

process = cms.Process("Refitting")

### Standard Configurations
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff") 
#process.load('Configuration.Geometry.GeometryRecoDB_cff')
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") 

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# choose!
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_GRun', '')
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_GRun', '')


### Track refitter specific stuff
process.load("RecoTracker.TrackProducer.TrackRefitter_cfi")
process.TrackRefitter.NavigationSchool = ''
process.TrackRefitter.Fitter = 'FlexibleKFFittingSmoother'

process.source = cms.Source ("PoolSource",
                             fileNames=cms.untracked.vstring('file:pickevents_1.root',
                            ),
                             skipEvents=cms.untracked.uint32(0)
                             )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.Path = cms.Path(process.TrackRefitter)

