import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
#from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START42_V17::All'
#process.GlobalTag.globaltag = 'MC_38Y_V14::All'

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
#process.ak5L1JPTOffset.era = 'Jec11V0'
#process.ak5L1JPTOffset.useCondDB = False

process.ak5L1JPTOffset.offsetService = cms.string('')

process.load('RecoJets.Configuration.RecoPFJets_cff')
process.kt6PFJets.doRhoFastjet = True
process.ak5PFJets.doAreaFastjet = True

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

'file:/afs/cern.ch/user/k/kovitang/scratch0/60A53D23-A1BB-E011-987B-001A92971B90.root'
    )

)

process.fileSaver = cms.EDAnalyzer("JetFileSaver",
                        OutputFile = cms.untracked.string('histo.root')
)

## Test for corrected jets - available only for 
#process.prefer("L2L3CorJetIC5Calo")

#process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeCone5CaloJets"),
#    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
#)


## AK5 Corrected jets
#process.JetAnalyzerAK5Cor = cms.EDAnalyzer("CaloJetTester",
#    src = cms.InputTag('ak5CaloJetsL2L3'),
#    JetCorrectionService = cms.string('ak5CaloJetsL2L3'),
    ##src = cms.InputTag("L2L3CorJetAK5Calo"),
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
	              *process.kt6PFJets 
                      #* process.ak5PFJets 
                      #--- Non-Standard sequence (that involve Producers)
                      #*process.ak5CaloJetsL2L3
 #                     *process.ZSPJetCorrectionsIcone5
 #                     *process.ZSPJetCorrectionsAntiKt5
 #                     *process.JetPlusTrackCorrectionsIcone5
 #                     *process.JetPlusTrackCorrectionsAntiKt5
#                      *process.JetAnalyzerAK5Cor
#                      *process.JetAnalyzerIC5JPT
#                      *process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation
                      #--- DQM stats module
#                      *process.dqmStoreStats
)
process.DQM.collectorHost = ''

