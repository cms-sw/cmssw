import FWCore.ParameterSet.Config as cms

## muonAssociatorByHits using only digiSimLinks (and TrackingParticles),
## not accessing the PSimHits directly. Useful if you run on RECOSIM without RAWSIM

from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters
muonAssociatorByHitsNoSimHitsHelper = cms.EDProducer("MuonToTrackingParticleAssociatorEDProducer",
    muonAssociatorByHitsCommonParameters
)
# don't read simhits, they're not there
muonAssociatorByHitsNoSimHitsHelper.CSCsimHitsTag = ""
muonAssociatorByHitsNoSimHitsHelper.RPCsimhitsTag = ""
muonAssociatorByHitsNoSimHitsHelper.GEMsimhitsTag = ""
muonAssociatorByHitsNoSimHitsHelper.DTsimhitsTag  = ""

### The following was used when running only on RECO
# don't normalize on the total number of hits (which is unknown, if I don't have simHits)
#muonAssociatorByHitsNoSimHitsHelper.AbsoluteNumberOfHits_muon = True
#muonAssociatorByHitsNoSimHitsHelper.AbsoluteNumberOfHits_track = True
#
### currently this is dealt with in the code itself (MuonAssociatorByHitsHelper.cc) 
### to allow ranking the simToReco matches according to the number of shared hits: 
### this is relevant for the definition of duplicates
   
# use only muon system
muonAssociatorByHitsNoSimHitsHelper.UseTracker = False

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( muonAssociatorByHitsNoSimHitsHelper, useGEMs = cms.bool(True) )
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( muonAssociatorByHitsNoSimHitsHelper, usePhase2Tracker = cms.bool(True) )
phase2_tracker.toModify( muonAssociatorByHitsNoSimHitsHelper, pixelSimLinkSrc = "simSiPixelDigis:Pixel" )
