import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

#process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer09_cff")

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/FCDDD13A-76C7-DE11-84F9-000423D985E4.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/80E5C510-7BC7-DE11-8EF8-001D09F24399.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/80CF2558-7BC7-DE11-9CA8-001D09F29146.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/42BBAB22-7DC7-DE11-9A5B-0030487A1990.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/3417B81C-79C7-DE11-A3A2-001D09F252DA.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/32145DFE-BFC7-DE11-8C4A-0030487C6062.root',
        '/store/relval/CMSSW_3_3_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v1/0005/02675539-74C7-DE11-97C5-001D09F24F1F.root'

    )

)

process.fileSaver = cms.EDFilter("JetFileSaver",
                                 OutputFile = cms.untracked.string('histo.root')
)

## Test for corrected jets - available only for 
process.prefer("L2L3JetCorrectorIC5Calo")

process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
)

## IC5 Corrected jets
process.JetAnalyzerIC5Cor = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("L2L3CorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

## IC5 JPT jets
process.JetAnalyzerIC5JPT = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

## AntiKt5 JPT jets
process.JetAnalyzerAk5JPT = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    srcGen = cms.InputTag("ak5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

process.p1 = cms.Path(process.fileSaver
                      #--- Non-Standard sequence (that involve Producers)
                      *process.L2L3CorJetIcone5
                      *process.ZSPJetCorrectionsIcone5
                      *process.ZSPJetCorrectionsAntiKt5
                      *process.JetPlusTrackCorrectionsIcone5
                      *process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerIC5Cor
                      *process.JetAnalyzerIC5JPT
                      *process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation)
                      #--- DQM stats module
                      #*process.dqmStoreStats)

process.DQM.collectorHost = ''

