import FWCore.ParameterSet.Config as cms

genTausFromZsForEmbeddingKineReweight = cms.EDProducer("GenParticlesFromZsSelectorForMCEmbedding",
    src = cms.InputTag("genParticles"),
    pdgIdsMothers = cms.vint32(23, 22),
    pdgIdsDaughters = cms.vint32(15),
    maxDaughters = cms.int32(2),
    minDaughters = cms.int32(2),
    before_or_afterFSR = cms.string("afterFSR")
)
genZdecayToTausForEmbeddingKineReweight = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('charge = 0'),
    decay = cms.string("genTausFromZsForEmbeddingKineReweight@+ genTausFromZsForEmbeddingKineReweight@-")
)

embeddingKineReweightGENembedding = cms.EDProducer("EmbeddingKineReweightProducer",
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_genEmbedding_mutau.root"),
    lutNames = cms.PSet(
        genTau2PtVsGenTau1Pt = cms.string('embeddingKineReweight_muon2Pt_vs_muon1Pt'),
        genTau2EtaVsGenTau1Eta = cms.string('embeddingKineReweight_muon2Eta_vs_muon1Eta'),                                               
        genDiTauMassVsGenDiTauPt = cms.string('embeddingKineReweight_diMuonMass_vs_diMuonPt')
    ),                                       
    srcGenDiTaus = cms.InputTag('genZdecayToTausForEmbeddingKineReweight'),
    minWeight = cms.double(0.),
    maxWeight = cms.double(1.e+1),
    verbosity = cms.int32(0)
)

embeddingKineReweightRECembedding = embeddingKineReweightGENembedding.clone(
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_recEmbedding_mutau.root"),
    lutNames = cms.PSet(
        genTau2PtVsGenTau1Pt = cms.string('embeddingKineReweight_muon2Pt_vs_muon1Pt'),
        genTau2EtaVsGenTau1Eta = cms.string('embeddingKineReweight_muon2Eta_vs_muon1Eta'),                                               
        genDiTauMassVsGenDiTauPt = cms.string('embeddingKineReweight_diMuonMass_vs_diMuonPt')
    ),       
    verbosity = cms.int32(0)
)    

embeddingKineReweightSequenceGENembedding = cms.Sequence(
    genTausFromZsForEmbeddingKineReweight
   + genZdecayToTausForEmbeddingKineReweight
   + embeddingKineReweightGENembedding
)

embeddingKineReweightSequenceRECembedding = cms.Sequence(
    genTausFromZsForEmbeddingKineReweight
   + genZdecayToTausForEmbeddingKineReweight
   + embeddingKineReweightRECembedding
)

embeddingKineReweightSequence = cms.Sequence(
    embeddingKineReweightSequenceGENembedding
   + embeddingKineReweightSequenceRECembedding
)
    


