# The following comments couldn't be translated into the new config version:
#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

whichJets  = "ak4PFJetsCHS"
applyJEC = True
corrLabel = 'ak4PFCHS'
tag =  'MCRUN2_74_V7::All'
useTrigger = False
triggerPath = "HLT_PFJet80_v*"
runOnMC    = True
#Flavour plots for MC: "all" = plots for all jets ; "dusg" = plots for d, u, s, dus, g independently ; not mandatory and any combinations are possible                                     
#b, c, light (dusg), non-identified (NI), PU jets plots are always produced
flavPlots = "allbcldusg"

###prints###
print "jet collcetion asked : ", whichJets
print "JEC applied?", applyJEC, ", correction:", corrLabel 
print "trigger will be used ? : ", useTrigger, ", Trigger paths:", triggerPath
print "is it MC ? : ", runOnMC, ", Flavours:", flavPlots
print "Global Tag : ", tag
############

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load("JetMETCorrections.Configuration.JetCorrectors_cff")
process.load("CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi")
process.load("RecoJets.JetAssociationProducers.ak4JTA_cff")
process.load("RecoBTag.Configuration.RecoBTag_cff")
process.load("PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi")
process.load("PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi")
process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")
process.JECseq = cms.Sequence(getattr(process,corrLabel+"L1FastL2L3CorrectorChain"))

newjetID=cms.InputTag(whichJets)
process.ak4JetFlavourInfos.jets = newjetID
if not "ak4PFJetsCHS" in whichJets:
    process.ak4JetTracksAssociatorAtVertexPF.jets = newjetID
    process.pfImpactParameterTagInfos.jets        = newjetID
    process.softPFMuonsTagInfos.jets              = newjetID
    process.softPFElectronsTagInfos.jets          = newjetID
    process.patJetGenJetMatch.src                 = newjetID

process.btagging = cms.Sequence(process.legacyBTagging + process.pfBTagging)
process.btagSequence = cms.Sequence(
    process.ak4JetTracksAssociatorAtVertexPF *
    process.btagging
    )
process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.btagSequence)

###
print "inputTag : ", process.ak4JetTracksAssociatorAtVertexPF.jets
###

process.load("Validation.RecoB.bTagAnalysis_firststep_cfi")
if runOnMC:
    process.flavourSeq = cms.Sequence(
        process.selectedHadronsAndPartons *
        process.ak4JetFlavourInfos
        )
    process.bTagValidationFirstStep.jetMCSrc = 'ak4JetFlavourInfos'
    process.bTagValidationFirstStep.applyPtHatWeight = False
    process.bTagValidationFirstStep.doJetID = True
    process.bTagValidationFirstStep.doJEC = applyJEC
    process.bTagValidation.JECsourceMC = cms.InputTag(corrLabel+"L1FastL2L3Corrector")
    process.bTagValidationFirstStep.flavPlots = flavPlots
    #process.bTagValidationFirstStep.ptRecJetMin = cms.double(20.)
    process.bTagValidationFirstStep.genJetsMatched = cms.InputTag("patJetGenJetMatch")
    process.bTagValidationFirstStep.doPUid = cms.bool(True)
    process.ak4GenJetsForPUid = cms.EDFilter("GenJetSelector",
                                             src = cms.InputTag("ak4GenJets"),
                                             cut = cms.string('pt > 8.'),
                                             filter = cms.bool(False)
                                             )
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi")
    process.patJetGenJetMatch.matched = cms.InputTag("ak4GenJetsForPUid")
    process.patJetGenJetMatch.maxDeltaR = cms.double(0.25)
    process.patJetGenJetMatch.resolveAmbiguities = cms.bool(True)
else:
    process.bTagValidationFirstStepData.doJEC = applyJEC
    process.bTagAnalysis.JECsourceData = cms.InputTag(corrLabel+"L1FastL2L3ResidualCorrector")
    process.JECseq *= (getattr(process,corrLabel+"ResidualCorrector") * getattr(process,corrLabel+"L1FastL2L3ResidualCorrector"))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.EDM = cms.OutputModule("DQMRootOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      "keep *_MEtoEDMConverter_*_*"),
                               fileName = cms.untracked.string('MEtoEDMConverter.root')
                               )
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
if useTrigger: 
    process.bTagHLT  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_PFJet40_v*"])
    process.bTagHLT.HLTPaths = [triggerPath]

if runOnMC:
    process.dqmSeq = cms.Sequence(process.ak4GenJetsForPUid * process.patJetGenJetMatch * process.flavourSeq * process.bTagValidationFirstStep)
else:
    process.dqmSeq = cms.Sequence(process.bTagValidationFirstStepData)

if useTrigger:
    process.plots = cms.Path(process.bTagHLT * process.JECseq * process.jetSequences * process.dqmSeq)
else:
    process.plots = cms.Path(process.JECseq * process.jetSequences * process.dqmSeq)
    
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

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = tag

