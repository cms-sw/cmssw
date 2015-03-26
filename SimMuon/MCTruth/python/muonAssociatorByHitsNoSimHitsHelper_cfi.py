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
muonAssociatorByHitsNoSimHitsHelper.DTsimhitsTag  = ""

### The following is useful when running only on RECO
# don't normalize on the total number of hits (which is unknown, if I don't have simHits)
muonAssociatorByHitsNoSimHitsHelper.AbsoluteNumberOfHits_muon = True
muonAssociatorByHitsNoSimHitsHelper.AbsoluteNumberOfHits_track = True
# use only muon system
muonAssociatorByHitsNoSimHitsHelper.UseTracker = False
