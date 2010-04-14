import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GlobalTag.globaltag = 'START3X_V26::All'
#process.GlobalTag.globaltag = 'GR_R_35X_V6::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FED8673E-F53D-DF11-9E58-0026189437EB.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEBF7874-EF3D-DF11-910D-002354EF3BDF.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEA8ECD8-F13D-DF11-8EBD-00304867BFAE.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE838E9F-F43D-DF11-BEBA-00261894393B.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE7D760E-F43D-DF11-878A-00304867BED8.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE2D63AD-F43D-DF11-B2B8-00261894395C.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC95A7F1-F13D-DF11-8C91-003048678C9A.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC5F5CA1-F53D-DF11-AFEE-002618FDA211.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC140D7E-F43D-DF11-B6C2-0026189437ED.root',
   )
)


################## DEDX ANALYSIS SEQUENCE MODULES ##################
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.TrajectoryInEvent = cms.bool(True)
process.TrackRefitter.src               = 'generalTracks'

from CondCore.DBCommon.CondDBCommon_cfi import *
process.MipsMap = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(  cms.PSet(record = cms.string('SiStripDeDxMip_3D_Rcd'),    tag =cms.string('MC7TeV_Deco_3D_Rcd_35X'))    )
)
process.MipsMap.connect = 'sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db'
process.MipsMap.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.es_prefer_geom=cms.ESPrefer("PoolDBESSource","MipsMap")

########################################################################

process.TFileService = cms.Service("TFileService", 
       	fileName = cms.string('out.root')
)

# OUT
process.OUT = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('out.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.p = cms.Path(process.TrackRefitter*process.HSCPTreeBuilderSeq)

#process.outpath  = cms.EndPath(process.OUT)
#process.schedule = cms.Schedule(process.p, process.outpath)
