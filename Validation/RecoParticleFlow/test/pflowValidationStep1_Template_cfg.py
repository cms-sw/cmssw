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
                            #,SkipEvent = cms.untracked.vstring('ProductNotFound')
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

#dataset = 'QCD'
#dataset = 'TTbar'
#dataset = 'ZEE'
#dataset = 'SingleElPt35'
dataset = 'DATA' # to be used with RunPFVal.sh 

#---------------------------------------
# List File names here
#---------------------------------------
if dataset == 'QCD' :
    sourceFiles = cms.untracked.vstring(    
#        '/store/relval/CMSSW_7_0_0_pre11/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/START70_V4_OldEG_FastSim-v1/00000/F0132E22-A76A-E311-BE6D-0025905A60B6.root',
      #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/EE4D1086-0F25-E311-BD19-003048678A6C.root',
      # '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/CCA45DB9-1325-E311-B521-0025905938AA.root',
      # '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/686ECB84-1025-E311-95BF-00261894387E.root',
      #  '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/6666ACEF-0F25-E311-B2CE-00248C0BE01E.root',
      #  '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/4E0FDB84-1025-E311-90B7-0026189438E0.root',
      #  '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/483621D2-1025-E311-BB70-003048B95B30.root',
      # '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/48118597-1125-E311-8645-003048678B7C.root'
        '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/1EB9CDDA-C8FE-E311-9082-0025905A6066.root'
        )
elif dataset == 'TTbar' :
    sourceFiles = cms.untracked.vstring( 
        #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/1A4CBAAC-48EC-E211-8A00-001E67397EB8.root',
        #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/24366CE4-42EC-E211-A3B9-003048D4988C.root',
        #'/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/8476F595-44EC-E211-B8E7-003048F00B16.root'
        'file:/afs/cern.ch/user/l/lecriste/miniAOD_validation_sequence_multiThreaded/CMSSW_7_2_0_pre3/src/20.0_SingleMuPt10+SingleMuPt10+DIGI+RECO+HARVEST/patTuple_mini.root'
        )
elif dataset == 'ZEE' :
    sourceFiles = cms.untracked.vstring( 
        '/store/relval/CMSSW_7_1_0_pre7/RelValZEE/GEN-SIM-RECO/PRE_STA71_V3-v1/00000/6EAA0A95-68D1-E311-AD8E-0026189438B9.root'
#        '/store/relval/CMSSW_7_0_0_pre4/RelValZEE/GEN-SIM-RECO/PU_PRE_ST62_V8-v1/00000/1CF2D144-6824-E311-85D5-003048678DD6.root'
        )
elif dataset == 'SingleElPt35' :
    sourceFiles = cms.untracked.vstring( 
        '/store/relval/CMSSW_6_2_0/RelValSingleElectronPt35/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/E05AE099-56EC-E211-9FF8-003048CF9B0C.root'
        )

process.PoolSource.fileNames = sourceFiles;

#-------------
# Global Tag
#-------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup'] #'mc'

## for 6_2_0 QCD
#process.GlobalTag.globaltag = 'PRE_ST62_V8::All'
## for 6_2_0 TTbar
#process.GlobalTag.globaltag = 'PRE_ST62_V8::v3'

#--------------------------------------------------
# Core DQM Stuff and definition of output EDM file
#--------------------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.EDM = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_PFlowDQM"),
                               fileName = cms.untracked.string('MEtoEDM_'+dataset+'_PFlow.root')
)

#--------------------------------------------------
# PFElectron Specific
#--------------------------------------------------
# start code moved to python/PFElectronValidation.cff.py
#process.pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
#                                      pdgId = cms.vint32(11, -11),
#                                      #src = cms.InputTag("pfNoPileUp")
#                                     src = cms.InputTag("particleFlow")
#                                      )

#process.gensource = cms.EDProducer("GenParticlePruner",
#                                   src = cms.InputTag("genParticles"),
#                                   select = cms.vstring('drop *',
#                                                        # for matching
#                                                        #'keep+ pdgId = 23',
#                                                        'keep pdgId = 11', 
#                                                        'keep pdgId = -11' 
#                                                        ## for fake rate
#                                                        #'keep pdgId = 211', # pi+
#                                                        #'keep pdgId = -211' # pi-
#                                                        )
#                                   )
# end code moved to python/PFElectronValidation.cff.py

process.pfPileUp = cms.EDProducer("PFPileUp",
                                  Enable = cms.bool(True),
                                  PFCandidates = cms.InputTag("particleFlow"),
                                  verbose = cms.untracked.bool(False),
                                  #Vertices = cms.InputTag("offlinePrimaryVertices")
                                  Vertices = cms.InputTag("offlinePrimaryVerticesWithBS")
                                  )

#process.pfNoPileUp = cms.EDProducer("TPPileUpPFCandidatesOnPFCandidates",
#                                    bottomCollection = cms.InputTag("particleFlow"),
#                                    enable = cms.bool(True),
#                                    topCollection = cms.InputTag("pfPileUp"),
#                                    name = cms.untracked.string('pileUpOnPFCandidates'),
#                                    verbose = cms.untracked.bool(False)
#                                    )

process.pfElectronBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
                                            maxDeltaPhi = cms.double(0.5),
                                            BenchmarkLabel = cms.string('PFlowElectrons'),
                                            OnlyTwoJets = cms.bool(False),
                                            maxEta = cms.double(2.5),
                                            minEta = cms.double(-1),
                                            recPt = cms.double(2.0),
                                            minDeltaPhi = cms.double(-0.5),
                                            PlotAgainstRecoQuantities = cms.bool(False),
                                            minDeltaEt = cms.double(-100.0),
                                            OutputFile = cms.untracked.string('benchmark.root'),
                                            StartFromGen = cms.bool(False),
                                            deltaRMax = cms.double(0.05),
                                            maxDeltaEt = cms.double(50.0),
                                            InputTruthLabel = cms.InputTag("gensource"),
                                            InputRecoLabel = cms.InputTag("pfAllElectrons"),
                                            doMetPlots = cms.bool(False)
                                            )

# start code moved to python/PFElectronValidation.cff.py
#process.pfElectronSequence = cms.Sequence(
#    #process.pfPileUp + 
#    #process.pfNoPileUp + 
#    process.pfAllElectrons + 
#    process.gensource
#    )
# end code moved to python/PFElectronValidation.cff.py

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
                                   process.DisplayEventContent,
                                   #outputCommands = cms.untracked.vstring('keep *'),
                                   fileName = cms.untracked.string('display.root')
                                   )

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
                                process.PFTau)
                                
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

#process.load("RecoParticleFlow.PFProducer.particleFlowCandidateChecker_cfi")
##process.particleFlowCandidateChecker.pfCandidatesReco = cms.InputTag("particleFlow","","REPROD")
##process.particleFlowCandidateChecker.pfCandidatesReReco = cms.InputTag("particleFlow","","REPROD2")
##process.particleFlowCandidateChecker.pfJetsReco = cms.InputTag("ak5PFJets","","REPROD")
##process.particleFlowCandidateChecker.pfJetsReReco = cms.InputTag("ak5PFJets","","REPROD2")

process.pfReRecoSequence = cms.Sequence(
    process.localReReco
    + process.globalReReco
    + process.pfReReco
    + process.genReReco
    #+ process.particleFlowCandidateChecker
    )


#--------------------------------------------
# PFDQM modules to book/fill actual histograms
#----------------------------------------------
process.load("Validation.RecoParticleFlow.PFJetValidation_cff")

process.load("Validation.RecoParticleFlow.PFMETValidation_cff")
process.load("Validation.RecoParticleFlow.PFJetResValidation_cff")

process.load("Validation.RecoParticleFlow.PFElectronValidation_cff") 
#process.load("Validation.RecoParticleFlow.PFMuonValidation_cff")
process.load("Validation.RecoMET.METRelValForDQM_cff")
process.load("Validation.RecoParticleFlow.miniAODDQM_cff")

# The complete reprocessing
process.p = cms.Path(
#    process.pfElectronSequence +
#    process.pfReRecoSequence +  # not needed for global validation used in case of software development
    #process.pfJetValidationSequence +
    #process.pfMETValidationSequence +
    #process.pfJetResValidationSequence +
    #process.METValidation +
#    process.pfMuonValidationSequence +  
    #process.pfElectronValidationSequence +
#    process.pfElectronBenchmarkGeneric + # replaced by pfElectronBenchmarkGeneric
    process.miniAODDQMSequence +
    process.MEtoEDMConverter
    )

process.outpath = cms.EndPath(
    process.EDM
    #+ process.reco 
    #+ process.display
    )

