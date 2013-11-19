import FWCore.ParameterSet.Config as cms


from CommonTools.ParticleFlow.pfNoPileUp_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons
patMuons.addGenMatch = cms.bool(False)
patMuons.embedHighLevelSelection = cms.bool(True)
patMuons.usePV = cms.bool(False)

patMuons.embedCaloMETMuonCorrs = cms.bool(False) # avoid crash
patMuons.embedTcMETMuonCorrs = cms.bool(False)
# No need to swich off following
#patMuons.embedTpfmsMuon = cms.bool(True),
#    embedPFCandidate = cms.bool(True),
#    embedStandAloneMuon = cms.bool(True),
#    embedCombinedMuon = cms.bool(True),
#    embedPickyMuon = cms.bool(True)


goodVertex = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("(!isFake) & ndof > 3 & abs(z) < 15 & position.Rho < 2"),
    filter = cms.bool(True)
)


muonsWithPFIso = cms.EDProducer("MuonWithPFIsoProducerCopy",
         MuonTag = cms.untracked.InputTag("muons")
       , PfTag = cms.untracked.InputTag("pfNoPileUp")
       , UsePfMuonsOnly = cms.untracked.bool(False)
       , TrackIsoVeto = cms.untracked.double(0.01)
       , GammaIsoVeto = cms.untracked.double(0.07)
       , NeutralHadronIsoVeto = cms.untracked.double(0.1)
)

patMuons.muonSource = cms.InputTag("muonsWithPFIso")

goodMuons = cms.EDFilter("PATMuonSelector",
  src = cms.InputTag("patMuons"),
     cut = cms.string(
           'pt > 10 & abs(eta) < 2.5 & isGlobalMuon & isTrackerMuon ' \
                 + ' & innerTrack.hitPattern.numberOfValidTrackerHits > 10 & innerTrack.hitPattern.numberOfValidPixelHits > 0' \
                 + ' & abs(dB)<0.2 & globalTrack.normalizedChi2 < 10' \
                 + ' & globalTrack.hitPattern.numberOfValidMuonHits > 0 & numberOfMatches > 1'
  ),
  filter = cms.bool(True)
)



goodMuonsPFIso =cms.EDFilter("PATMuonSelector",
  src = cms.InputTag("goodMuons"),
  #cut = cms.string("iso03.sumPt/pt < 0.1 "  ),
  cut = cms.string("trackIso() < 0.1*pt "  ),
  #cut = cms.string("trackIso() < 11110.1*pt "  ),
  filter = cms.bool(False)
)



goldenZmumuCandidatesGe0IsoMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    # require one of the muons with pT > 20
    cut = cms.string('charge = 0 & max(daughter(0).pt,daughter(1).pt)>20'),
    decay = cms.string("goodMuons@+ goodMuons@-")
)


goldenZmumuCandidatesGe1IsoMuons = goldenZmumuCandidatesGe0IsoMuons.clone()
goldenZmumuCandidatesGe1IsoMuons.decay = cms.string("goodMuons@+ goodMuonsPFIso@-")

goldenZmumuCandidatesGe2IsoMuons = goldenZmumuCandidatesGe0IsoMuons.clone()
goldenZmumuCandidatesGe2IsoMuons.decay = cms.string("goodMuonsPFIso@+ goodMuonsPFIso@-")


goldenZmumuFilter = cms.EDFilter("CandViewCountFilter",
    #src = cms.InputTag("goldenZmumuCandidatesGe0IsoMuons"), # loose selection 
    src = cms.InputTag("goldenZmumuCandidatesGe1IsoMuons"),  # tight selection                            
    minNumber = cms.uint32(1)
)

print "Zmumu skim will use: ", goldenZmumuFilter.src



goldenZmumuSelectionSequence = cms.Sequence(
  goodVertex
  * pfNoPileUpSequence 
  * muonsWithPFIso
  * patMuons 
  * goodMuons
  * goodMuonsPFIso 
  * goldenZmumuCandidatesGe0IsoMuons 
  * goldenZmumuCandidatesGe1IsoMuons
  * goldenZmumuCandidatesGe2IsoMuons
  * goldenZmumuFilter
)
