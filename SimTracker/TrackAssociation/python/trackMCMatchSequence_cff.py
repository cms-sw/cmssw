import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackMCMatch_cfi import *
from SimTracker.TrackAssociation.standAloneMuonsMCMatch_cfi import *
from SimTracker.TrackAssociation.globalMuonsMCMatch_cfi import *
from SimTracker.TrackAssociation.allTrackMCMatch_cfi import *
trackMCMatchSequence = cms.Sequence(trackMCMatch*standAloneMuonsMCMatch*globalMuonsMCMatch*allTrackMCMatch)

