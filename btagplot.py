# The following comments couldn't be translated into the new config version:
#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

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
process.GlobalTag.globaltag = 'POSTLS162_V1::All'

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMOffline.RecoB.bTagSequences_cff")
process.load("Validation.RecoB.bTagAnalysis_cfi")
process.bTagValidation.tagConfig = [cms.PSet(
            parameters = cms.PSet(
                                discriminatorStart = cms.double(-0.05),
                                discriminatorEnd = cms.double(1.05),
                                nBinEffPur = cms.int32(200),
                                # the constant b-efficiency for the differential plots versus pt and eta
                                effBConst = cms.double(0.5),
                                endEffPur = cms.double(1.005),
                                startEffPur = cms.double(-0.005)
                                ),
            label = cms.InputTag("combinedSecondaryVertexBJetTags"),
            folder = cms.string("CSVnew")
),
cms.PSet(
            parameters = cms.PSet(
                                discriminatorStart = cms.double(-0.05),
                                discriminatorEnd = cms.double(1.05),
                                nBinEffPur = cms.int32(200),
                                # the constant b-efficiency for the differential plots versus pt and eta
                                effBConst = cms.double(0.5),
                                endEffPur = cms.double(1.005),
                                startEffPur = cms.double(-0.005)
                                ),
            label = cms.InputTag("combinedSecondaryVertexBJetTags","","RECO"),
            folder = cms.string("CSVorig")
),


]

newjetID="ak4PFJetsCHS"
process.myak4JetTracksAssociatorAtVertex.jets = newjetID
process.softPFMuonsTagInfos.jets             = newjetID
process.softPFElectronsTagInfos.jets          = newjetID
process.AK4byRef.jets                         = newjetID

process.bTagValidation.jetMCSrc = 'AK4byValAlgo'
process.bTagValidation.allHistograms = True 
process.bTagValidation.ptRanges = cms.vdouble(50.0, 80.0, 120.0,300,500,1000)

process.bTagValidation.applyPtHatWeight = False
#process.bTagValidation.flavPlots = "allbcl" #if contains "noall" plots for all jets not booked, if contains "dusg" all histograms booked, default : all, b, c, udsg, ni
process.bTagValidation.flavPlots = "dusg" #if contains "noall" plots for all jets not booked, if contains "dusg" all histograms booked, default : all, b, c, udsg, ni

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)
#process.myPartons.src = "prunedGenParticles"
#   process.jetSequences = cms.Sequence(process.goodOfflinePrimaryVertices * process.btagSequence)
process.dqmSeq = cms.Sequence(process.flavourSeq* process.bTagValidation * process.dqmSaver)
process.plots = cms.Path(process.dqmSeq)
    
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [
#file:btag001nominalQuality.root"
"file:trk_00.root",
"file:trk_01.root",
"file:trk_02.root",
"file:trk_03.root",
"file:trk_04.root",
"file:trk_05.root",
"file:trk_06.root",
"file:trk_07.root",
"file:trk_08.root",
"file:trk_09.root",
"file:trk_10.root",
"file:trk_11.root",
"file:trk_12.root",
"file:trk_13.root",
"file:trk_14.root"
]

