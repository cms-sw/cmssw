import FWCore.ParameterSet.Config as cms

from TauAnalysis.MCEmbeddingTools.genMuonRadCorrAnalyzer_cfi import genMuonRadCorrAnalyzer
muonRadiationCorrWeightProducer = cms.EDProducer("MuonRadiationCorrWeightProducer",
    inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/muonRadiationCorrWeightProducer.root"),
    srcMuonsBeforeRad = cms.InputTag('generator', 'muonsBeforeRad'),
    srcMuonsAfterRad = cms.InputTag('generator', 'muonsAfterRad'),
    lutDirectoryRef = cms.string('genMuonRadCorrAnalyzerPHOTOS'), # NOTE: needs to match 'applyMuonRadiationCorrection' parameter of ParticleReplacerZtautau used in embedding (default = PHOTOS)
    lutDirectoryOthers = cms.PSet(                              
        pythia = cms.string('genMuonRadCorrAnalyzer')
    ),
    binningMuonEn = genMuonRadCorrAnalyzer.binningMuonEn,
    minWeight = cms.double(0.),
    maxWeight = cms.double(2.),
    verbosity = cms.int32(0)                                                   
)
