import FWCore.ParameterSet.Config as cms

## muonAssociatorByHits using only digiSimLinks (and TrackingParticles),
## not accessing the PSimHits directly. Useful if you run on RECOSIM without RAWSIM

from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters
muonAssociatorByHitsESProducerNoSimHits = cms.ESProducer("MuonAssociatorESProducer",
    muonAssociatorByHitsCommonParameters,
    ComponentName = cms.string("muonAssociatorByHits_NoSimHits"),
)
# don't read simhits, they're not there
muonAssociatorByHitsESProducerNoSimHits.CSCsimHitsTag = ""
muonAssociatorByHitsESProducerNoSimHits.RPCsimhitsTag = ""
muonAssociatorByHitsESProducerNoSimHits.DTsimhitsTag  = ""
muonAssociatorByHitsESProducerNoSimHits.GEMsimhitsTag  = ""

### The following is useful when running only on RECO
# don't normalize on the total number of hits (which is unknown, if I don't have simHits)
muonAssociatorByHitsESProducerNoSimHits.AbsoluteNumberOfHits_muon = True
muonAssociatorByHitsESProducerNoSimHits.AbsoluteNumberOfHits_track = True
# use only muon system
muonAssociatorByHitsESProducerNoSimHits.UseTracker = False
