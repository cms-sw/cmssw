import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackHistory_cff import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ic5JetVetoedTracksAssociatorAtVertex = cms.EDFilter("JetVetoedTracksAssociatorAtVertex",
    trackHistory,
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)


