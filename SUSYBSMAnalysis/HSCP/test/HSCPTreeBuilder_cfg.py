import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = 'START3X_V26::All'
#process.GlobalTag.globaltag = 'GR_R_35X_V6::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_10_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_11_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_12_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_13_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_15_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_16_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_17_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_19_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_22_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_27_2.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_28_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_29_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_2_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_30_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_31_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_33_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_34_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_35_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_36_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_37_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_38_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_39_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_3_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_40_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_41_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_43_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_44_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_46_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_48_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_4_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_50_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_51_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_55_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_56_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_57_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_5_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_6_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_7_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_8_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_9_1.root',

   )
)


################## DEDX ANALYSIS SEQUENCE MODULES ##################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilderFromSkim_cff")

from CondCore.DBCommon.CondDBCommon_cfi import *
process.MipsMap = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    appendToDataLabel = cms.string(''),
#    toGet = cms.VPSet(  cms.PSet(record = cms.string('SiStripDeDxMip_3D_Rcd'),    tag =cms.string('MC7TeV_Deco_3D_Rcd_35X'))    )
    toGet = cms.VPSet(  cms.PSet(record = cms.string('SiStripDeDxMip_3D_Rcd'),    tag =cms.string('Data7TeV_Deco_3D_Rcd_35X'))    )
)
#process.MipsMap.connect = 'sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db'
process.MipsMap.connect = 'sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db'
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

#In case you start from the HSCP Skim, you can produce the tree directly
process.p = cms.Path(process.HSCPTreeBuilderSeq)
#In case you don't start form the HSCP Skim, better to first skims tracks with very low Pt.
#process.p = cms.Path(process.exoticaHSCPSeq + process.HSCPTreeBuilderSeq)


#process.outpath  = cms.EndPath(process.OUT)
#process.schedule = cms.Schedule(process.p, process.outpath)
