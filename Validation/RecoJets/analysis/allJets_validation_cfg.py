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
#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

#process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
#process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer09_cff")

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/FA30B22D-290E-DF11-9FE6-0030487C5CFA.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/E8BA16A6-290E-DF11-918A-0030487A18F2.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/E6FE3BB3-2D0E-DF11-8FE5-0030487C5CFA.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/DCE3F7FD-2D0E-DF11-BD5E-003048D37580.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/B621A1D0-2B0E-DF11-9085-0030487A3C92.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/AE40A5A5-540E-DF11-A48E-003048D2C1C4.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/A2009204-320E-DF11-B0F7-0030487CAEAC.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/8EA79A6D-280E-DF11-B934-0030487C778E.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/7807E87C-2C0E-DF11-B1A6-0030487CD14E.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/72084001-280E-DF11-ADC9-0030487CAF5E.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/508E1207-2D0E-DF11-ABF6-0030487CD6DA.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/40BB2A62-290E-DF11-B2BF-0030487CD13A.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_3XY_V20-v1/0008/14FD1882-2D0E-DF11-9AA2-0030487C5CFA.root'

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

process.p1 = cms.Path(process.fileSaver
                      #--- Non-Standard sequence (that involve Producers)
                      *process.L2L3CorJetIcone5
                      #*process.ZSPJetCorrectionsIcone5
                      #*process.ZSPJetCorrectionsAntiKt5
                      #*process.JetPlusTrackCorrectionsIcone5
                      #*process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerIC5Cor
                      #*process.JetAnalyzerIC5JPT
                      #*process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation)
                      #--- DQM stats module
                      #*process.dqmStoreStats)

process.DQM.collectorHost = ''

