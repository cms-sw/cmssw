# test file for PFDQM Validation
# performs a Jet and MET Validations (PF vs Gen and PF vs Calo)
# creates an EDM file with histograms filled with PFCandidate data,
# present in the PFJetMonitor and PFMETMonitor classes in DQMOffline/PFTau
# package for matched PFCandidates. Matching histograms (delta pt etc)
# are also available. 
import FWCore.ParameterSet.Config as cms
process = cms.Process("PFlowDQM")
#------------------------
# Message Logger Settings
#------------------------
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#--------------------------------------
# Event Source & # of Events to process
#---------------------------------------
process.source = cms.Source("PoolSource",
                   fileNames = cms.untracked.vstring()
                 )
process.maxEvents = cms.untracked.PSet(
                      input = cms.untracked.int32(-1)
                    )
#-------------
# Global Tag
#-------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_42_V3::All'
#--------------------------------------------------
# Core DQM Stuff and definition of output EDM file
#--------------------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.EDM = cms.OutputModule("PoolOutputModule",
                outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_PFlowDQM"),
               fileName = cms.untracked.string('MEtoEDM_PFlow.root')
)
#--------------------------------------------------
# PFElectron Specific
#--------------------------------------------------
process.pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
                                 pdgId = cms.vint32(11, -11),
                                 src = cms.InputTag("pfNoPileUp")
)

process.gensource = cms.EDProducer("GenParticlePruner",
              src = cms.InputTag("genParticles"),
              select = cms.vstring('drop *',
                     'keep pdgId = 11',
                     'keep pdgId = -11')
)
process.pfNoPileUp = cms.EDProducer("TPPileUpPFCandidatesOnPFCandidates",
                                        bottomCollection = cms.InputTag("particleFlow"),
                                        enable = cms.bool(True),
                                        topCollection = cms.InputTag("pfPileUp"),
                                        name = cms.untracked.string('pileUpOnPFCandidates'),
                                        verbose = cms.untracked.bool(False)
)
process.pfPileUp = cms.EDProducer("PFPileUp",
                                      Enable = cms.bool(True),
                                      PFCandidates = cms.InputTag("particleFlow"),
                                      verbose = cms.untracked.bool(False),
                                      Vertices = cms.InputTag("offlinePrimaryVertices")
)
process.pfElectronSequence = cms.Sequence(process.pfPileUp+process.pfNoPileUp+process.pfAllElectrons+process.gensource)
#--------------------------------------------
# PFDQM modules to book/fill actual histograms
#----------------------------------------------
process.load("Validation.RecoParticleFlow.PFJetValidation_cff")
process.load("Validation.RecoParticleFlow.PFMETValidation_cff")
process.load("Validation.RecoParticleFlow.PFElectronValidation_cff")


process.p =cms.Path(
    process.pfElectronSequence  +  
    process.pfJetValidationSequence +
    process.pfMETValidationSequence +
    process.pfElectronValidationSequence +    
    process.MEtoEDMConverter
    )

process.outpath = cms.EndPath(process.EDM)
#--------------------------------------
# List File names here
#---------------------------------------
process.PoolSource.fileNames = [
    '/store/relval/CMSSW_4_2_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_42_V3-v1/0008/AE8048D3-3C3C-E011-9696-00304867BFAE.root'
]

