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
    input = cms.untracked.int32(-1)
    )

#dataset = 'QCD'
#dataset = 'TTbar'
#dataset = 'ZEE'
#dataset = 'SingleElPt35'
dataset = 'QCD' # to be used with RunPFVal.sh 

#---------------------------------------
# List File names here
#---------------------------------------
if dataset == 'QCD' :
    sourceFiles = cms.untracked.vstring(    
        #'/store/relval/CMSSW_4_2_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_42_V3-v1/0008/AE8048D3-3C3C-E011-9696-00304867BFAE.root'
        #'/store/relval/CMSSW_6_2_0_pre6_patch1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/36AD630A-D4BE-E211-9F2E-003048678B86.root'
        #'file:/afs/cern.ch/user/l/lecriste/muons/CMSSW_6_2_0_pre6_patch1/src/RecoParticleFlow/Configuration/test/reco.root'
        
        #'/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/604E213E-49EC-E211-9D8D-003048F0E5CE.root',
        #'/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/68DC246F-56EC-E211-B2E5-003048CF94A4.root'

       # '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/0ABFBC60-0525-E311-8407-00261894397E.root' # missing pfAllElectrons

        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/EE4D1086-0F25-E311-BD19-003048678A6C.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/CCA45DB9-1325-E311-B521-0025905938AA.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/686ECB84-1025-E311-95BF-00261894387E.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/6666ACEF-0F25-E311-B2CE-00248C0BE01E.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/4E0FDB84-1025-E311-90B7-0026189438E0.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/483621D2-1025-E311-BB70-003048B95B30.root',
        '/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000HS/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/48118597-1125-E311-8645-003048678B7C.root'

        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/0279C975-4F25-E311-94BD-0025905964BC.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/0E2EE868-4F25-E311-AD57-00261894380A.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/3838CC69-4F25-E311-BC80-003048678FD6.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/3C109169-4F25-E311-8F96-003048678FB4.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/580CAA6A-4F25-E311-91B0-0026189438FD.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/58AF6334-9725-E311-BABF-003048FFD7D4.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/5C62B66D-4F25-E311-86EC-0026189438BA.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/7851306B-4F25-E311-8365-002618943800.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/8E4B3277-4F25-E311-9A3F-002590596486.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/9051CF68-4F25-E311-8898-002618943864.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/A478D26B-4F25-E311-9680-0026189438F5.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/AA0F9E6C-4F25-E311-A96C-003048678BB2.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/BAEAEC69-4F25-E311-8C25-003048678FD6.root',
        #'/store/relval/CMSSW_7_0_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/CE89016B-4F25-E311-9F27-003048679166.root'
        )
elif dataset == 'TTbar' :
    sourceFiles = cms.untracked.vstring( 
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/1A4CBAAC-48EC-E211-8A00-001E67397EB8.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/24366CE4-42EC-E211-A3B9-003048D4988C.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/8476F595-44EC-E211-B8E7-003048F00B16.root'
        )
elif dataset == 'ZEE' :
    sourceFiles = cms.untracked.vstring( 
        #'/store/relval/CMSSW_6_2_0/RelValZEE/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/50169250-40EC-E211-964F-02163E008C2D.root',
        #'/store/relval/CMSSW_6_2_0/RelValZEE/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/D656BB27-4CEC-E211-807F-003048D4988C.root'
        '/store/relval/CMSSW_7_0_0_pre4/RelValZEE/GEN-SIM-RECO/PU_PRE_ST62_V8-v1/00000/1CF2D144-6824-E311-85D5-003048678DD6.root'
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
process.pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
                                      pdgId = cms.vint32(11, -11),
                                      #src = cms.InputTag("pfNoPileUp")
                                      src = cms.InputTag("particleFlow")
                                      )

process.gensource = cms.EDProducer("GenParticlePruner",
                                   src = cms.InputTag("genParticles"),
                                   select = cms.vstring('drop *',
                                                        # for matching
                                                        #'keep+ pdgId = 23',
                                                        'keep pdgId = 11', 
                                                        'keep pdgId = -11' 
                                                        ## for fake rate
                                                        #'keep pdgId = 211', # pi+
                                                        #'keep pdgId = -211' # pi-
                                                        )
                                   )

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

process.pfElectronSequence = cms.Sequence(
    #process.pfPileUp + 
    #process.pfNoPileUp + 
    process.pfAllElectrons + 
    process.gensource
    )

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

process.load("RecoParticleFlow.PFProducer.particleFlowCandidateChecker_cfi")
#process.particleFlowCandidateChecker.pfCandidatesReco = cms.InputTag("particleFlow","","REPROD")
#process.particleFlowCandidateChecker.pfCandidatesReReco = cms.InputTag("particleFlow","","REPROD2")
#process.particleFlowCandidateChecker.pfJetsReco = cms.InputTag("ak5PFJets","","REPROD")
#process.particleFlowCandidateChecker.pfJetsReReco = cms.InputTag("ak5PFJets","","REPROD2")

process.pfReRecoSequence = cms.Sequence(
    process.localReReco
    + process.globalReReco
    + process.pfReReco
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
process.load("Validation.RecoParticleFlow.PFJetResValidation_cff")
process.pfJetResValidation1.SkimParameter.switchOn = cms.bool(True)
process.pfJetResValidation2.SkimParameter.switchOn = cms.bool(True)

process.load("Validation.RecoParticleFlow.PFElectronValidation_cff") 
# needed to run Muon validation: it need tag V00-10-01 not yet in CMSSW_6_2_0
#process.load("Validation.RecoParticleFlow.PFMuonValidation_cff")

# The complete reprocessing
process.p = cms.Path(
    process.pfElectronSequence +
#    process.pfReRecoSequence +  # not needed for global validation used in case of software development
    process.pfJetValidationSequence +
    process.pfMETValidationSequence +
    process.pfJetResValidationSequence +
#    process.pfMuonValidationSequence +  
    process.pfElectronValidationSequence +
#    process.pfElectronBenchmarkGeneric + # replaced by pfElectronBenchmarkGeneric
    process.MEtoEDMConverter
    )

process.outpath = cms.EndPath(
    process.EDM
    #+ process.reco 
    #+ process.display
    )

