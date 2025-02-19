import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.GlobalTag.globaltag = 'START3X_V26::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
   )
)


########################################################################
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

process.generalTracksSkim.filter       = cms.bool(False) 
process.HSCParticleProducer.filter     = cms.bool(False)
process.HSCPTreeBuilder.reccordGenInfo = cms.untracked.bool(True)

################## DEDX ANALYSIS SEQUENCE MODULES ##################
from CondCore.DBCommon.CondDBCommon_cfi import *
process.MipsMap = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(  cms.PSet(record = cms.string('SiStripDeDxMip_3D_Rcd'),    tag =cms.string('MC7TeV_Deco_3D_Rcd_35X'))    )
#   toGet = cms.VPSet(  cms.PSet(record = cms.string('SiStripDeDxMip_3D_Rcd'),    tag =cms.string('Data7TeV_Deco_3D_Rcd_35X'))    )
)
process.MipsMap.connect = 'sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db'
#process.MipsMap.connect = 'sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db'
process.MipsMap.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.es_prefer_geom=cms.ESPrefer("PoolDBESSource","MipsMap")

process.dedxCNPHarm2.calibrationPath = cms.string("file:Gains.root")
process.dedxCNPTru40.calibrationPath = cms.string("file:Gains.root")
process.dedxCNPMed.calibrationPath   = cms.string("file:Gains.root")
process.dedxProd.calibrationPath     = cms.string("file:Gains.root")
process.dedxSmi.calibrationPath      = cms.string("file:Gains.root")
process.dedxASmi.calibrationPath     = cms.string("file:Gains.root")
########################################################################

#process.TFileService = cms.Service("TFileService", 
#        fileName = cms.string('HSCP_tree.root')
#)

process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep *_csc2DRecHits_*_*",
         "keep *_cscSegments_*_*",
         "keep *_dt1DRecHits_*_*",
         "keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_reducedHSCPhbhereco_*_*",
         "keep *_reducedHSCPEcalRecHitsEB_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_standAloneMuons_*_*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep *_HSCParticleProducer_*_*",
    ),
    fileName = cms.untracked.string('HSCP.root')
)

########################################################################

process.p = cms.Path(process.genParticles + process.exoticaHSCPSeq + process.HSCParticleProducerSeq)

process.outpath  = cms.EndPath(process.OUT)
process.schedule = cms.Schedule(process.p, process.outpath)
