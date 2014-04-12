import FWCore.ParameterSet.Config as cms

ZmumuEvtSelEffCorrWeightProducer = cms.EDProducer("ZmumuEvtSelEffCorrWeightProducer",
    selectedMuons = cms.InputTag(''), # CV: replaced in embeddingCustomizeAll.py                                                
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/ZmumuEvtSelEffCorrWeightLUTs.root"),
    lutEfficiencyPt = cms.string("ZmumuEvtSelEff_muMinusPt_vs_muPlusPt"),
    lutEffCorrEta = cms.string("ZmumuEvtSelEffCorr_muMinusEta_vs_muPlusEta"),
    minWeight = cms.double(0.),
    maxWeight = cms.double(10.),
    verbosity = cms.int32(0)                                                   
)
