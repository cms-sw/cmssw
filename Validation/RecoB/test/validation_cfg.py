# The following comments couldn't be translated into the new config version:
#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')

options.register ('jets',
                  "ak5PFCHS", # default value, allowed : "ak5PF", "ak5PFCHS"
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,  
                  "jet collection to use")

options.parseArguments()

whichJets  = options.jets 
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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = tag

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMOffline.RecoB.bTagSequences_cff")
#process.bTagHLT.HLTPaths = ["HLT_PFJet80_v*"] #uncomment this line if you want to use different trigger

process.JECAlgo = cms.Sequence(process.ak5JetsJEC * process.PFJetsFilter)
newjetID=cms.InputTag("PFJetsFilter")
process.myak5JetTracksAssociatorAtVertex.jets = newjetID
process.softPFMuonsTagInfos.jets              = newjetID
process.softPFElectronsTagInfos.jets          = newjetID
process.AK5byRef.jets                         = newjetID
if whichJets=="ak5PF":
    process.ak5JetsJEC.src = 'ak5PFJets'
    process.ak5JetsJEC.correctors = ['ak5PFL1L2L3']
###
print "inputTag : ", process.myak5JetTracksAssociatorAtVertex.jets
###

if runOnMC:
    process.load("Validation.RecoB.bTagAnalysis_cfi")
    process.bTagValidation.jetMCSrc = 'AK5byValAlgo'
    process.bTagValidation.allHistograms = True 
    process.bTagValidation.applyPtHatWeight = False
    process.bTagValidation.flavPlots = "allbcl" #if contains "noall" plots for all jets not booked, if contains "dusg" all histograms booked, default : all, b, c, udsg, ni
else :
    process.ak5JetsJEC.correctors[0] += 'Residual'
    process.load("DQMOffline.RecoB.bTagAnalysisData_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.JECAlgo * process.btagSequence)

if runOnMC:
    process.dqmSeq = cms.Sequence(process.flavourSeq * process.bTagValidation * process.dqmSaver)
else:
    process.dqmSeq = cms.Sequence(process.bTagAnalysis * process.dqmSaver)

if useTrigger:
    process.plots = cms.Path(process.bTagHLT * process.jetSequences * process.dqmSeq)
else:
    process.plots = cms.Path(process.jetSequences * process.dqmSeq)
    
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [

]

