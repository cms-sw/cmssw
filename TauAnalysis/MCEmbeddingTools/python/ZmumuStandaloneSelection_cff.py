import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons
patMuonsForZmumuSelection = patMuons.clone()
patMuonsForZmumuSelection.addGenMatch = cms.bool(False)
patMuonsForZmumuSelection.embedCaloMETMuonCorrs = cms.bool(False)
patMuonsForZmumuSelection.embedTcMETMuonCorrs = cms.bool(False)

goodVertex = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("(!isFake) & ndof >= 4 & abs(z) < 24 & position.Rho < 2"),
    filter = cms.bool(False)
)

goodMuons = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patMuonsForZmumuSelection"),
    cut = cms.string(
        'pt > 10 && abs(eta) < 2.5 && isGlobalMuon && isPFMuon '
        ' && track.hitPattern.trackerLayersWithMeasurement > 5 & innerTrack.hitPattern.numberOfValidPixelHits > 0'
        ' && abs(dB) < 0.2 && globalTrack.normalizedChi2 < 10'
        ' && globalTrack.hitPattern.numberOfValidMuonHits > 0 && numberOfMatchedStations > 1'
    ),
    filter = cms.bool(False)
)

highestPtMuPlus = cms.EDFilter("UniquePATMuonSelector",
    src = cms.VInputTag('goodMuons'),                           
    cut = cms.string('charge > 0.5'),
    rank = cms.string('pt'),
    filter = cms.bool(False)
)

highestPtMuMinus = cms.EDFilter("UniquePATMuonSelector",
    src = cms.VInputTag('goodMuons'),                           
    cut = cms.string('charge < -0.5'),
    rank = cms.string('pt'),
    filter = cms.bool(False)
)

goodMuonsPFIso = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("goodMuons"),
    cut = cms.string('pfIsolationR04().sumChargedHadronPt < (0.1*pt)'),
    filter = cms.bool(False)
)

highestPtMuPlusPFIso = cms.EDFilter("UniquePATMuonSelector",
    src = cms.VInputTag('goodMuonsPFIso'),                           
    cut = cms.string('charge > 0.5'),
    rank = cms.string('pt'),
    filter = cms.bool(False)
)

highestPtMuMinusPFIso = cms.EDFilter("UniquePATMuonSelector",
    src = cms.VInputTag('goodMuonsPFIso'),                           
    cut = cms.string('charge < -0.5'),
    rank = cms.string('pt'),
    filter = cms.bool(False)
)

goldenZmumuCandidatesGe0IsoMuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    # require one of the muons with pT > 20
    cut = cms.string('charge = 0 & max(daughter(0).pt, daughter(1).pt) > 20'),
    decay = cms.string("highestPtMuPlus@+ highestPtMuMinus@-")
)

goldenZmumuCandidatesGe1IsoMuonsComb1 = goldenZmumuCandidatesGe0IsoMuons.clone(
    decay = cms.string("highestPtMuPlusPFIso@+ highestPtMuMinus@-") # mu+ passes isolation
)
goldenZmumuCandidatesGe1IsoMuonsComb2 = goldenZmumuCandidatesGe1IsoMuonsComb1.clone(
    decay = cms.string("highestPtMuPlus@+ highestPtMuMinusPFIso@-") # mu- passes isolation
)
goldenZmumuCandidatesGe1IsoMuons = cms.EDFilter("UniqueCompositeCandidateSelector",
    src = cms.VInputTag(
        'goldenZmumuCandidatesGe1IsoMuonsComb1',
        'goldenZmumuCandidatesGe1IsoMuonsComb2'
    ),                           
    rank = cms.string('daughter(0).pt*daughter(1).pt'),
    filter = cms.bool(False)
)

goldenZmumuCandidatesGe2IsoMuons = goldenZmumuCandidatesGe0IsoMuons.clone(
    decay = cms.string("highestPtMuPlusPFIso@+ highestPtMuMinusPFIso@-")
)

goldenZmumuFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goldenZmumuCandidatesGe0IsoMuons"),  # loose selection 
    #src = cms.InputTag("goldenZmumuCandidatesGe1IsoMuons"), # medium-tight selection
    #src = cms.InputTag("goldenZmumuCandidatesGe2IsoMuons"), # tight selection                             
    minNumber = cms.uint32(1)
)

goldenZmumuPreFilterHistos = cms.EDAnalyzer("AcceptanceHistoProducer",
    dqmDir = cms.string('PreFilter'),
    srcGenParticles = cms.InputTag("genParticles")
)
goldenZmumuPostFilterHistos = goldenZmumuPreFilterHistos.clone(
    dqmDir = cms.string('PostFilter')
)
from DQMServices.Components.MEtoEDMConverter_cff import MEtoEDMConverter
MEtoEDMConverter.deleteAfterCopy = cms.untracked.bool(False)

goldenZmumuSelectionSequence = cms.Sequence(
    goodVertex
   + patMuonsForZmumuSelection 
   + goodMuons
   + highestPtMuPlus
   + highestPtMuMinus 
   + goodMuonsPFIso
   + highestPtMuPlusPFIso
   + highestPtMuMinusPFIso 
   + goldenZmumuCandidatesGe0IsoMuons
   + goldenZmumuCandidatesGe1IsoMuonsComb1
   + goldenZmumuCandidatesGe1IsoMuonsComb2
   + goldenZmumuCandidatesGe1IsoMuons
   + goldenZmumuCandidatesGe2IsoMuons
   + goldenZmumuPreFilterHistos
)

goldenZmumuFilterSequence = cms.Sequence(
    goldenZmumuSelectionSequence    
   + goldenZmumuFilter
   + goldenZmumuPostFilterHistos
   + MEtoEDMConverter
)
