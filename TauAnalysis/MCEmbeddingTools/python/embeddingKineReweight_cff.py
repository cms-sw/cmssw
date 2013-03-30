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

# CV: define correction for muon Pt smearing introduced by muon -> muon + photon radiation corrections
embeddingKineReweightGENtoEmbedded = cms.EDProducer("EmbeddingKineReweightProducer",
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoEmbedded.root"),
    lutNames = cms.PSet(
        genDiTauMassVsGenDiTauPt = cms.string('embeddingKineReweight_diMuonMass_vs_diMuonPt'),
        genTau2PtVsGenTau1Pt = cms.string('embeddingKineReweight_muon2Pt_vs_muon1Pt')
    ),                                       
    srcGenDiTaus = cms.InputTag('genZdecayToTausForEmbeddingKineReweight'),
    minWeight = cms.double(0.),
    maxWeight = cms.double(1.e+1),
    verbosity = cms.int32(0)
)

# CV: define correction for muon Pt smearing caused by track (mis)reconstruction
embeddingKineReweightGENtoREC = embeddingKineReweightGENtoEmbedded.clone(
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoREC.root"),
    lutNames = cms.PSet(
        genDiTauMassVsGenDiTauPt = cms.string('embeddingKineReweight_diMuonMass_vs_diMuonPt'),
        genTau2PtVsGenTau1Pt = cms.string('embeddingKineReweight_muon2Pt_vs_muon1Pt')
    ),       
    verbosity = cms.int32(0)
)    

embeddingKineReweightSequenceGENtoREC = cms.Sequence(
    genTausFromZsForEmbeddingKineReweight
   + genZdecayToTausForEmbeddingKineReweight
   + embeddingKineReweightGENtoREC
)

embeddingKineReweightSequenceGENtoEmbedded = cms.Sequence(
    genTausFromZsForEmbeddingKineReweight
   + genZdecayToTausForEmbeddingKineReweight
   + embeddingKineReweightGENtoEmbedded
)

embeddingKineReweightSequence = cms.Sequence(
    embeddingKineReweightSequenceGENtoREC
   + embeddingKineReweightSequenceGENtoEmbedded
)
    


