# The following comments couldn't be translated into the new config version:
#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

whichJets  = "ak5PFCHS" # default value, allowed : "ak5PF", "ak5PFCHS", add "NoJEC" to run the code with no JEC applied
useTrigger = False
runOnMC    = True
tag =  'START70_V4::All'

###prints###
print "jet collcetion asked : ", whichJets
print "trigger will be used ? : ", useTrigger
print "is it MC ? : ", runOnMC
print "Global Tag : ", tag
############

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff") #old one, to use for old releases
process.load("Configuration.Geometry.GeometryIdeal_cff")
#process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = tag

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMOffline.RecoB.bTagSequences_cff")
#bTagHLT.HLTPaths = ["HLT_PFJet80_v*"] #uncomment this line if you want to use different trigger

newjetID=cms.InputTag("ak5PFJetsCHS")
process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.btagSequence)
if "NoJEC" in whichJets and not "CHS" in whichJets : newjetID=cms.InputTag("ak5PFJets")
if not "NoJEC" in whichJets:
    process.JECAlgo = cms.Sequence(process.ak5JetsJEC * process.PFJetsFilter)
    process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.JECAlgo * process.btagSequence)
    newjetID=cms.InputTag("PFJetsFilter")
    if whichJets=="ak5PF":
        process.ak5JetsJEC.src = 'ak5PFJets'
        process.ak5JetsJEC.correctors = ['ak5PFL1FastL2L3']
process.myak5JetTracksAssociatorAtVertex.jets = newjetID
process.softPFMuonsTagInfos.jets              = newjetID
process.softPFElectronsTagInfos.jets          = newjetID
process.AK5byRef.jets                         = newjetID

###                                                                                                                                                                                                     
print "inputTag : ", process.myak5JetTracksAssociatorAtVertex.jets
###   

process.load("Validation.RecoB.bTagAnalysis_firststep_cfi")
if runOnMC:
    process.bTagValidationFirstStep.jetMCSrc = 'AK5byValAlgo'
    process.bTagValidationFirstStep.allHistograms = True
    process.bTagValidationFirstStep.applyPtHatWeight = False
    process.bTagValidationFirstStep.flavPlots = "allbcl" #if contains "noall" plots for all jets not booked, if contains "dusg" all histograms booked, default : all, b, c, udsg, ni
    #process.bTagValidation.ptRecJetMin = cms.double(20.)                                                                                          
    process.bTagValidation.genJetsMatched = cms.InputTag("patJetGenJetMatch")
    process.bTagValidation.doPUid = cms.bool(True)
    process.ak5GenJetsForPUid = cms.EDFilter("GenJetSelector",
                                             src = cms.InputTag("ak5GenJets"),
                                             cut = cms.string('pt > 8.'),
                                             filter = cms.bool(False)
                                             )
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi")
    process.patJetGenJetMatch.src = newjetID
    process.patJetGenJetMatch.matched = cms.InputTag("ak5GenJetsForPUid")
    process.patJetGenJetMatch.maxDeltaR = cms.double(0.25)
    process.patJetGenJetMatch.resolveAmbiguities = cms.bool(True)
else :
    process.ak5JetsJEC.correctors[0] += 'Residual'
                                  
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.EDM = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      "keep *_MEtoEDMConverter_*_*"),
                               fileName = cms.untracked.string('MEtoEDMConverter.root')
                               )
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.JECAlgo * process.btagSequence)

if runOnMC:
    process.dqmSeq = cms.Sequence(process.ak5GenJetsForPUid * process.patJetGenJetMatch * process.flavourSeq * process.bTagValidationFirstStep * process.MEtoEDMConverter)
else:
    process.dqmSeq = cms.Sequence(bTagValidationFirstStepData * process.MEtoEDMConverter)
    
if useTrigger:
    process.plots = cms.Path(process.bTagHLT * process.jetSequences * process.dqmSeq)
else:
    process.plots = cms.Path(process.jetSequences * process.dqmSeq)
                                              
process.outpath = cms.EndPath(process.EDM)

process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [

]

