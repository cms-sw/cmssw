# The following comments couldn't be translated into the new config version:
#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

whichJets  = "ak5PF"
useTrigger = False
runOnMC    = True
tag =  'START60_V1::All'

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
#bTagHLT.HLTPaths = ["HLT_PFJet80_v*"] #uncomment this line if you want to use different trigger

if whichJets=="ak5PFnoPU":
    process.out = cms.OutputModule("PoolOutputModule",
                                   outputCommands = cms.untracked.vstring('drop *'),
                                   fileName = cms.untracked.string('EmptyFile.root')
                                   )
    process.load("PhysicsTools.PatAlgos.patSequences_cff")
    from PhysicsTools.PatAlgos.tools.pfTools import *
    postfix="PF2PAT"
    usePF2PAT(process,runPF2PAT=True, jetAlgo="AK5", runOnMC=runOnMC, postfix=postfix)
    applyPostfix(process,"patJetCorrFactors",postfix).payload = cms.string('AK5PFchs')
    process.pfPileUpPF2PAT.Vertices = cms.InputTag('goodOfflinePrimaryVertices')
    process.pfPileUpPF2PAT.checkClosestZVertex = cms.bool(False)
    from DQMOffline.RecoB.bTagSequences_cff import JetCut
    process.selectedPatJetsPF2PAT.cut = JetCut
    process.JECAlgo = cms.Sequence( getattr(process,"patPF2PATSequence"+postfix) )
    newjetID=cms.InputTag("selectedPatJetsPF2PAT")
elif whichJets=="ak5PFJEC":
    process.JECAlgo = cms.Sequence(process.ak5PFJetsJEC * process.PFJetsFilter)
    newjetID=cms.InputTag("PFJetsFilter")
    
if not whichJets=="ak5PF":
    process.myak5JetTracksAssociatorAtVertex.jets = newjetID
    process.softMuonTagInfos.jets                 = newjetID
    process.softElectronTagInfos.jets             = newjetID
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

if whichJets=="ak5PF":
    process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.btagSequence)
else:
    process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.JECAlgo * process.btagSequence)
    
if runOnMC:
    process.dqmSeq = cms.Sequence(process.flavourSeq * process.bTagValidationFirstStep * process.MEtoEDMConverter)
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

