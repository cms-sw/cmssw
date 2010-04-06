import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
#from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

#process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
#
#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

# check # of bins
#process.load("DQMServices.Components.DQMStoreStats_cfi")

#process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
#process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/FAE9A792-493E-DF11-BE5F-001A92810AA0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/F80FDE51-E03D-DF11-87B6-0018F3D095EA.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/E67A00DF-E13D-DF11-9D8A-0018F3D096E0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/ACDCC124-E53D-DF11-AACF-001A92971B80.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/A4F4A0CA-E13D-DF11-8E48-003048678B08.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/70C9A1E3-E03D-DF11-A369-001A928116B0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0010/1C5C38E4-E23D-DF11-96EF-003048679274.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V3-v1/0009/EA146DEB-D93D-DF11-9E86-003048678A76.root'

    )

)

process.fileSaver = cms.EDFilter("JetFileSaver",
                                 OutputFile = cms.untracked.string('histo.root')
)

## Test for corrected jets - available only for 
#process.prefer("L2L3CorJetIC5Calo")

#process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeCone5CaloJets"),
#    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
#)


## AK5 Corrected jets
process.JetAnalyzerAK5Cor = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("L2L3CorJetAK5Calo"),
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


### IC5 JPT jets
#JetAnalyzerIC5JPT = cms.EDFilter("CaloJetTester",
#    src = cms.InputTag("ic5JPTJetsL2L3"),
#    srcGen = cms.InputTag("iterativeCone5GenJets"),
##    TurnOnEverything = cms.untracked.string('yes'),
##    TurnOnEverything = cms.untracked.string('no'),
##    outputFile = cms.untracked.string('histo.root'),
##    outputFile = cms.untracked.string('test.root'),
#    genEnergyFractionThreshold = cms.double(0.05),
#    genPtThreshold = cms.double(1.0),
#    RThreshold = cms.double(0.3),
#    reverseEnergyFractionThreshold = cms.double(0.5)
#)

### AntiKt5 JPT jets
#JetAnalyzerAk5JPT = cms.EDFilter("CaloJetTester",
#    src = cms.InputTag("ak5JPTJetsL2L3"),
#    srcGen = cms.InputTag("ak5GenJets"),
##    TurnOnEverything = cms.untracked.string('yes'),
##    TurnOnEverything = cms.untracked.string('no'),
##    outputFile = cms.untracked.string('histo.root'),
##    outputFile = cms.untracked.string('test.root'),
#    genEnergyFractionThreshold = cms.double(0.05),
#    genPtThreshold = cms.double(1.0),
#    RThreshold = cms.double(0.3),
#    reverseEnergyFractionThreshold = cms.double(0.5)
#)

process.p1 = cms.Path(process.fileSaver
                      #--- Non-Standard sequence (that involve Producers)
                      *process.L2L3CorJetAK5Calo
 #                     *process.ZSPJetCorrectionsIcone5
 #                     *process.ZSPJetCorrectionsAntiKt5
 #                     *process.JetPlusTrackCorrectionsIcone5
 #                     *process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerAK5Cor
#                      *process.JetAnalyzerIC5JPT
#                      *process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation
                      #--- DQM stats module
#                      *process.dqmStoreStats
)
process.DQM.collectorHost = ''

