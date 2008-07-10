import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Magnetic Field
from Configuration.StandardSequences.MagneticField_cff import *

# Track Associators
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *

generalGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
    trackAssociator = cms.untracked.string('TrackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("generalTracks"),
    genParticles = cms.untracked.InputTag("genParticles")
)

globalMuonsGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
    trackAssociator = cms.untracked.string('TrackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("globalMuons"),
    genParticles = cms.untracked.InputTag("genParticles")
)

standAloneMuonsGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
    trackAssociator = cms.untracked.string('TrackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("standAloneMuons"),
    genParticles = cms.untracked.InputTag("genParticles")
)

genTrackMatcher = cms.Sequence(generalGenTrackMatcher*globalMuonsGenTrackMatcher*standAloneMuonsGenTrackMatcher)


