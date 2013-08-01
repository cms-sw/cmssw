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
                            fileNames = cms.untracked.vstring(),
                            eventsToProcess = cms.untracked.VEventRange(),
                            secondaryFileNames = cms.untracked.vstring(),
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            skipEvents = cms.untracked.uint32(0)
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
    )
#-------------
# Global Tag
#-------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")


#process.GlobalTag.globaltag = 'MC_42_V3::All'

from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['startup'] #'mc'

## for 6_2_0_pre6_patch1
#process.GlobalTag.globaltag = 'START61_V11::All'
#process.GlobalTag.globaltag = 'PRE_ST62_V6::All'
#process.GlobalTag.globaltag = 'PRE_ST62_V6::v1' #No "HcalPFCorrsRcd" record found in the EventSetup. Please add an ESSource or ESProducer that delivers such a record.

## for 6_2_0 QCD
process.GlobalTag.globaltag = 'PRE_ST62_V8::All'

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
#process.pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
#                                 pdgId = cms.vint32(11, -11),
#                                 src = cms.InputTag("pfNoPileUp")
#)
#
#process.gensource = cms.EDProducer("GenParticlePruner",
#              src = cms.InputTag("genParticles"),
#              select = cms.vstring('drop *',
#                     'keep pdgId = 11',
#                     'keep pdgId = -11')
#)
#process.pfNoPileUp = cms.EDProducer("TPPileUpPFCandidatesOnPFCandidates",
#                                        bottomCollection = cms.InputTag("particleFlow"),
#                                        enable = cms.bool(True),
#                                        topCollection = cms.InputTag("pfPileUp"),
#                                        name = cms.untracked.string('pileUpOnPFCandidates'),
#                                        verbose = cms.untracked.bool(False)
#)
#process.pfPileUp = cms.EDProducer("PFPileUp",
#                                      Enable = cms.bool(True),
#                                      PFCandidates = cms.InputTag("particleFlow"),
#                                      verbose = cms.untracked.bool(False),
#                                      Vertices = cms.InputTag("offlinePrimaryVertices")
#)
#process.pfElectronSequence = cms.Sequence(
#    process.pfPileUp + 
#    process.pfNoPileUp + 
#    process.pfAllElectrons + 
#    process.gensource
#    )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_cff")
#process.display = cms.OutputModule("PoolOutputModule",
#                                   process.DisplayEventContent,
#                                   #outputCommands = cms.untracked.vstring('keep *'),
#                                   fileName = cms.untracked.string('display.root')
#                                   )

process.load("Configuration.EventContent.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
                                process.RECOSIMEventContent,
                                fileName = cms.untracked.string('reco.root')
                                )

#process.particleFlowTmp.useHO = False

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.siPixelRecHits +
                                   process.siStripMatchedRecHits +
                                   #process.hbhereflag +
                                   process.particleFlowCluster +
                                   process.ecalClusters)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot +
                                     process.recopixelvertexing +
                                     process.ckftracks +
                                     process.caloTowersRec +
                                     process.vertexreco +
                                     process.recoJets +
                                     process.muonrecoComplete +
                                     process.muoncosmicreco +
                                     process.egammaGlobalReco +
                                     process.pfTrackingGlobalReco +
                                     process.egammaHighLevelRecoPrePF +
                                     process.muoncosmichighlevelreco +
                                     process.metreco)

# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco +
                                process.egammaHighLevelRecoPostPF +
                                process.muonshighlevelreco +
                                process.particleFlowLinks +
                                process.recoPFJets +
                                process.recoPFMET +
                                process.PFTau
                                )
                                
# Gen Info re-processing
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoMET.Configuration.GenMETParticles_cff")
process.load("RecoMET.Configuration.RecoGenMET_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("RecoParticleFlow.Configuration.HepMCCopy_cfi")
process.genReReco = cms.Sequence(process.generator +
                                 process.genParticles +
                                 process.genJetParticles +
                                 process.recoGenJets +
                                 process.genMETParticles +
                                 process.recoGenMET +
                                 process.particleFlowSimParticle
                                 )

process.load("RecoParticleFlow.PFProducer.particleFlowCandidateChecker_cfi")
#process.particleFlowCandidateChecker.pfCandidatesReco = cms.InputTag("particleFlow","","REPROD")
#process.particleFlowCandidateChecker.pfCandidatesReReco = cms.InputTag("particleFlow","","REPROD2")
#process.particleFlowCandidateChecker.pfJetsReco = cms.InputTag("ak5PFJets","","REPROD")
#process.particleFlowCandidateChecker.pfJetsReReco = cms.InputTag("ak5PFJets","","REPROD2")

process.pfDanieleSequence = cms.Sequence(
    process.localReReco +
    process.globalReReco +
    process.pfReReco
    + process.genReReco
    + process.particleFlowCandidateChecker
    )


#--------------------------------------------
# PFDQM modules to book/fill actual histograms
#----------------------------------------------
process.load("Validation.RecoParticleFlow.PFJetValidation_cff")
process.pfJetValidation1.SkimParameter.switchOn = cms.bool(True)
process.pfJetValidation2.SkimParameter.switchOn = cms.bool(True)
process.load("Validation.RecoParticleFlow.PFMETValidation_cff")
process.load("Validation.RecoParticleFlow.PFDanieleValidation_cff")
process.load("Validation.RecoParticleFlow.PFMuonValidation_cff")
#process.load("Validation.RecoParticleFlow.PFElectronValidation_cff")

# The complete reprocessing
process.p = cms.Path(
    process.pfDanieleSequence +  
    process.pfJetValidationSequence +
    process.pfMETValidationSequence +
    process.pfDanieleValidationSequence +
    process.pfMuonValidationSequence +  
    #process.pfElectronValidationSequence +
    process.MEtoEDMConverter
    )

#process.outpath = cms.EndPath(process.EDM)

process.outpath = cms.EndPath(
    process.EDM
    #+ process.reco 
    #+ process.display
    )

#---------------------------------------
# List File names here
#---------------------------------------
process.PoolSource.fileNames = [
    #'/store/relval/CMSSW_4_2_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_42_V3-v1/0008/AE8048D3-3C3C-E011-9696-00304867BFAE.root'
    #'/store/relval/CMSSW_6_2_0_pre6_patch1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/36AD630A-D4BE-E211-9F2E-003048678B86.root'
    #'file:/afs/cern.ch/user/l/lecriste/muons/CMSSW_6_2_0_pre6_patch1/src/RecoParticleFlow/Configuration/test/reco.root'
    # QCD
    '/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/604E213E-49EC-E211-9D8D-003048F0E5CE.root',
    '/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/68DC246F-56EC-E211-B2E5-003048CF94A4.root'
]

#process.Tracer = cms.Service("Tracer")
