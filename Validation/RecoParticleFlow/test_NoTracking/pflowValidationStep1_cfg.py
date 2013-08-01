# test file for PFDQM Validation
# performs a Jet and MET Validations (PF vs Gen and PF vs Calo)
# creates an EDM file with histograms filled with PFCandidate data,
# present in the PFJetMonitor and PFMETMonitor classes in DQMOffline/PFTau
# package for matched PFCandidates. Matching histograms (delta pt etc)
# are also available. 
import FWCore.ParameterSet.Config as cms
process = cms.Process("PFlowDQMnoTracking")
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
                            skipEvents = cms.untracked.uint32(5)
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

#process.GlobalTag.globaltag = 'START61_V11::All'
#process.GlobalTag.globaltag = 'PRE_ST62_V6::All'
#process.GlobalTag.globaltag = 'PRE_ST62_V6::v1' #No "HcalPFCorrsRcd" record found in the EventSetup. Please add an ESSource or ESProducer that delivers such a record.

## for 6_2_0 QCD
process.GlobalTag.globaltag = 'PRE_ST62_V8::All'
## for 6_2_0 TTbar
#process.GlobalTag.globaltag = 'PRE_ST62_V8::v3'

#--------------------------------------------------
# Core DQM Stuff and definition of output EDM file
#--------------------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.EDM = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_PFlowDQMnoTracking"),
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
#process.pfElectronSequence = cms.Sequence(process.pfPileUp +process.pfNoPileUp +process.pfAllElectrons +process.gensource)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_NoTracking_cff")
process.display = cms.OutputModule("PoolOutputModule",
                                   process.DisplayEventContent,
                                   fileName = cms.untracked.string('display.root')
                                   )

# modify reconstruction sequence
process.pfTrack.MuColl = cms.InputTag('muons')
process.particleFlowTmp.muons = cms.InputTag('muons')
process.particleFlow.FillMuonRefs = False
#process.particleFlowTmp.useHO = True


# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence( process.particleFlowCluster +
                                    process.ecalClusters
                                    )

process.globalReReco = cms.Sequence(process.particleFlowTrackWithDisplacedVertex +
                                    process.gsfEcalDrivenElectronSequence
                                    )
#process.globalReReco = cms.Sequence( process.egammaGlobalReco +
#                                     process.pfTrackingGlobalReco
#                                     process.particleFlowTrackWithDisplacedVertex +
#                                     process.egammaHighLevelRecoPrePF
#                                     )

# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco +
                                process.egammaHighLevelRecoPostPF +
                                #process.muonshighlevelreco +
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
    + process.particleFlowCandidateChecker_NoTracking
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
process.load("Validation.RecoParticleFlow.PFElectronValidation_cff")


# The complete reprocessing
process.p = cms.Path(
    process.pfDanieleSequence +
    process.pfJetValidationSequence +
    process.pfMETValidationSequence +
    process.pfDanieleValidationSequence_NoTracking +
    process.pfMuonValidationSequence +  
    process.pfElectronValidationSequence_NoTracking +
    process.MEtoEDMConverter
    )

#process.outpath = cms.EndPath(process.EDM)

process.outpath = cms.EndPath(
    process.EDM
    #+ process.display
)

#---------------------------------------
# List File names here
#---------------------------------------
process.PoolSource.fileNames = [
    #'/store/relval/CMSSW_4_2_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_42_V3-v1/0008/AE8048D3-3C3C-E011-9696-00304867BFAE.root'
    #'/store/relval/CMSSW_6_2_0_pre6_patch1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/36AD630A-D4BE-E211-9F2E-003048678B86.root'
    #'file:/afs/cern.ch/user/l/lecriste/muons/CMSSW_6_2_0_pre6_patch1/src/RecoParticleFlow/Configuration/test/display.root'
    #'file:/afs/cern.ch/user/l/lecriste/muons/CMSSW_6_2_0_pre6_patch1/src/Validation/RecoParticleFlow/test/reco.root'

    # QCD
    '/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/604E213E-49EC-E211-9D8D-003048F0E5CE.root',
    '/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/68DC246F-56EC-E211-B2E5-003048CF94A4.root'

    # TTbar
    #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/1A4CBAAC-48EC-E211-8A00-001E67397EB8.root',
    #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/24366CE4-42EC-E211-A3B9-003048D4988C.root',
    #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/8476F595-44EC-E211-B8E7-003048F00B16.root'

]

#process.Tracer = cms.Service("Tracer")
