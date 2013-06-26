import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackClassifier_cff import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *

ic5JetVetoedTracksAssociatorAtVertex = cms.EDProducer("JetVetoedTracksAssociatorAtVertex",
    trackClassifier,
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)


