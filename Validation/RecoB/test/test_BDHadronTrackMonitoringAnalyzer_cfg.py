import FWCore.ParameterSet.Config as cms

process = cms.Process('BDHadronTrackMonitorAnalyzerDQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)


process.PoolSource.fileNames = [
	'/store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/002033F7-36B9-E611-A30B-0025905A48EC.root'
    	#'root://xrootd-cms.infn.it//store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/002033F7-36B9-E611-A30B-0025905A48EC.root',
	#'root://xrootd-cms.infn.it//store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/00A6E94A-3DB9-E611-AAAD-0CC47A4D7604.root',
	#'root://xrootd-cms.infn.it//store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/00EB8925-41B9-E611-BF1E-0CC47A78A456.root',
	#'root://xrootd-cms.infn.it//store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/020F6DD1-36B9-E611-B893-0CC47A4D7604.root',
	#'root://xrootd-cms.infn.it//store/relval/CMSSW_8_1_0_pre16/RelValTTbar_13/GEN-SIM-RECODEBUG/PU25ns_81X_upgrade2017_realistic_v22_HS_rsb-v1/10000/027E621E-3FB9-E611-9415-0025905A6104.root'
	]


from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent, patEventContentNoCleaning
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT.root")
                                     )
                            
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')  #for MC
process.GlobalTag.globaltag = "80X_mcRun2_asymptotic_v4"

postfix = "PFlow"

from PhysicsTools.PatAlgos.tools.pfTools import *
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

patAlgosToolsTask = getPatAlgosToolsTask(process)

usePF2PAT(
            process,
            runPF2PAT=True,
            jetAlgo="AK4",
            runOnMC=True,
            postfix=postfix,
            jetCorrections=('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
            pvCollection=cms.InputTag('offlinePrimaryVertices')
)

 ## Top projections in PF2PAT
getattr(process,"pfPileUpJME"+postfix).checkClosestZVertex = False
getattr(process,"pfNoPileUpJME"+postfix).enable = True
getattr(process,"pfNoMuonJMEPFBRECO"+postfix).enable = False
getattr(process,"pfNoElectronJMEPFBRECO"+postfix).enable = False


# switch jet collection to make PAT collection
from PhysicsTools.PatAlgos.tools.jetTools import *
switchJetCollection(
        process,
        jetSource = cms.InputTag('ak4PFJetsCHS'),
        #'ak4PFJets'
        pfCandidates = cms.InputTag('particleFlow'),
        pvSource = cms.InputTag('offlinePrimaryVertices'),
        svSource = cms.InputTag('inclusiveCandidateSecondaryVertices'),
        muSource = cms.InputTag('muons'),
        elSource = cms.InputTag('gedGsfElectrons'),
        btagInfos = [
                    'pfImpactParameterTagInfos'
                    ,'pfSecondaryVertexTagInfos'
                    ,'pfInclusiveSecondaryVertexFinderTagInfos'
                    ,'pfSecondaryVertexNegativeTagInfos'
                    ,'pfInclusiveSecondaryVertexFinderNegativeTagInfos'
                    ,'softPFMuonsTagInfos'
                    ,'softPFElectronsTagInfos'
                    ,'pfInclusiveSecondaryVertexFinderCvsLTagInfos'
                    ,'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos'
        ],
        btagDiscriminators = ['pfJetBProbabilityBJetTags'
                            ,'pfJetProbabilityBJetTags'
                            ,'pfCombinedSecondaryVertexV2BJetTags'
        ],
        jetCorrections = ('AK4PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'], 'None'),
        genJetCollection = cms.InputTag('ak4GenJetsNoNu'),
        genParticles = cms.InputTag('genParticles'),
        postfix=postfix
)


## Add TagInfos to PAT jets
if hasattr(process,'patJets'+postfix) and getattr( getattr(process,'patJets'+postfix), 'addBTagInfo' ):
    setattr( getattr(process,'patJets'+postfix), 'addTagInfos', cms.bool(True) )


# my analyzer
process.load('Validation.RecoB.BDHadronTrackMonitoring_cfi')
process.BDHadronTrackMonitoringAnalyze.PatJetSource = cms.InputTag('selectedPatJets'+postfix)

process.load("SimTracker.TrackHistory.TrackHistory_cff")
process.load("SimTracker.TrackHistory.TrackClassifier_cff")
process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
patAlgosToolsTask.add(process.quickTrackAssociatorByHits)
patAlgosToolsTask.add(process.tpClusterProducer)

process.BDHadronTrackMonitoringAnalyzer = cms.Path(process.BDHadronTrackMonitoringAnalyze)
#process.dqmsave_step = cms.Path(process.DQMSaver)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)



# Schedule definition
process.schedule = cms.Schedule(
    process.BDHadronTrackMonitoringAnalyzer,
    process.DQMoutput_step,
#    process.dqmsave_step,
    tasks=[patAlgosToolsTask]
    )
