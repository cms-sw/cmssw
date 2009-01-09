import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoMuon.GlobalMuonMatchAnalyzer_cfi import *
TrackAssociatorByPosition.method = 'dist'
TrackAssociatorByPosition.MinIfNoMatch = True
TrackAssociatorByPosition.QCut = 10.


