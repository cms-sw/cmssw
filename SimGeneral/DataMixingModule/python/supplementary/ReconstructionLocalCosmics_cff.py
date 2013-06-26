import FWCore.ParameterSet.Config as cms

#
# tracker
#
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.Configuration.RecoTrackerBHM_cff import *
from RecoTracker.DeDx.dedxEstimators_Cosmics_cff import *


#
# calorimeters
#
from RecoLocalCalo.Configuration.RecoLocalCalo_Cosmics_cff import *
from RecoEcal.Configuration.RecoEcalCosmics_cff import *
#
# muons
#
from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
from RecoMuon.Configuration.RecoMuonCosmics_cff import *

# primary vertex
#from RecoVertex.Configuration.RecoVertexCosmicTracks_cff import *

# local reco
trackerCosmics = cms.Sequence(offlineBeamSpot*trackerlocalreco)
caloCosmics = cms.Sequence(calolocalreco)
muonsLocalRecoCosmics = cms.Sequence(muonlocalreco+muonlocalrecoNoDrift)

localReconstructionCosmics = cms.Sequence(trackerCosmics*caloCosmics*muonsLocalRecoCosmics)


reconstructionCosmics = cms.Sequence(localReconstructionCosmics)
