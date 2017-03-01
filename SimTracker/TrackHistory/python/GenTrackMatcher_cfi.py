import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Magnetic Field
from Configuration.StandardSequences.MagneticField_cff import *

# Track Associators
from SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi import *

generalGenTrackMatcher = cms.EDProducer("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.InputTag('trackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("generalTracks"),
    genParticles = cms.untracked.InputTag("genParticles")
)

globalMuonsGenTrackMatcher = cms.EDProducer("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.InputTag('trackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("globalMuons"),
    genParticles = cms.untracked.InputTag("genParticles")
)

standAloneMuonsGenTrackMatcher = cms.EDProducer("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.InputTag('trackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("standAloneMuons"),
    genParticles = cms.untracked.InputTag("genParticles")
)

genTrackMatcher = cms.Sequence(trackAssociatorByHits*generalGenTrackMatcher*globalMuonsGenTrackMatcher*standAloneMuonsGenTrackMatcher)


