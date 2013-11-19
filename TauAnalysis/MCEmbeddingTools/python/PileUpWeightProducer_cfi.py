import FWCore.ParameterSet.Config as cms

PileUpWeightProducerPUS10vs2012A = cms.EDProducer("PileUpWeightProducer",
    srcPileUpSummaryInfo = cms.InputTag('addPileupInfo'),
    sourceInputFile = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/Pileup_Summer12_53x.root"), # reweight from this distribution...
    targetInputFile = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/Pileup_2012A.root") # ... to this distribution
)
