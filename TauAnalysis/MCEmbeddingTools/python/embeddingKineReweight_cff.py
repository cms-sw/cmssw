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

embeddingKineReweightGENtoEmbedded = cms.EDProducer("EmbeddingKineReweightProducer",
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoEmbedded.root"),
    lutNames = cms.PSet(
        genDiTauPt = cms.string('embeddingKineReweight_genDiTauPt'),
        genDiTauMass = cms.string('embeddingKineReweight_genDiTauMass')
    ),                                       
    srcGenDiTaus = cms.InputTag('genZdecayToTausForEmbeddingKineReweight'),
    minWeight = cms.double(0.),
    maxWeight = cms.double(1.e+1),
    verbosity = cms.int32(0)
)

embeddingKineReweightGENtoREC = embeddingKineReweightGENtoEmbedded.clone(
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/makeEmbeddingKineReweightLUTs_GENtoREC.root")
)    

embeddingKineReweightSequence = cms.Sequence(
    genTausFromZsForEmbeddingKineReweight
   + genZdecayToTausForEmbeddingKineReweight
   + embeddingKineReweightGENtoEmbedded
   + embeddingKineReweightGENtoREC
)


