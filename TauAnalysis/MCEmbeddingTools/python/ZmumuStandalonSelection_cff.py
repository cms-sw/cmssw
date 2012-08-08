import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons
patMuons.addGenMatch = cms.bool(False)

goodVertex = cms.EDFilter("VertexSelector",
	src = cms.InputTag("offlinePrimaryVertices"),
	cut = cms.string("(!isFake) & ndof > 3 & abs(z) < 15 & position.Rho < 2"),
	filter = cms.bool(True)
)

goodMuons = cms.EDFilter("PATMuonSelector",
	src = cms.InputTag("patMuons"),
	cut = cms.string(
		'pt > 10 && abs(eta) < 2.5 && isGlobalMuon && isTrackerMuon '
		' && innerTrack.hitPattern.numberOfValidTrackerHits > 9 & innerTrack.hitPattern.numberOfValidPixelHits > 0'
		' && abs(dB) < 0.2 && globalTrack.normalizedChi2 < 10'
		' && globalTrack.hitPattern.numberOfValidMuonHits > 0 && numberOfMatches > 1'
	),
	filter = cms.bool(True)
)

goodMuonsPFIso = cms.EDFilter("PATMuonRefSelector",
	src = cms.InputTag("goodMuons"),
	cut = cms.string('pfIsolationR04().sumChargedHadronPt < 0.1*pt'),
	filter = cms.bool(False)
)

goldenZmumuCandidatesGe0IsoMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    # require one of the muons with pT > 20
    cut = cms.string('charge = 0 & max(daughter(0).pt,daughter(1).pt)>20'),
    decay = cms.string("goodMuons@+ goodMuons@-")
)

# Currently disabled as this produces duplicate muons; we don't need it anyway currently
#goldenZmumuCandidatesGe1IsoMuons = goldenZmumuCandidatesGe0IsoMuons.clone(
#	decay = cms.string("goodMuons@+ goodMuonsPFIso@-")
#)

goldenZmumuCandidatesGe2IsoMuons = goldenZmumuCandidatesGe0IsoMuons.clone(
	decay = cms.string("goodMuonsPFIso@+ goodMuonsPFIso@-")
)

goldenZmumuFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goldenZmumuCandidatesGe0IsoMuons"), # loose selection 
    #src = cms.InputTag("goldenZmumuCandidatesGe1IsoMuons"),  # tight selection                            
    minNumber = cms.uint32(1)
)

goldenZmumuSelectionSequence = cms.Sequence(
  goodVertex
  * patMuons 
  * goodMuons
  * goodMuonsPFIso 
  * goldenZmumuCandidatesGe0IsoMuons 
#  * goldenZmumuCandidatesGe1IsoMuons
  * goldenZmumuCandidatesGe2IsoMuons
  * goldenZmumuFilter
)
