import FWCore.ParameterSet.Config as cms

process = cms.Process("Prova")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
## GE1/1
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
##
## GE1/1 + GE2/1
#process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023_cff')
##
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")

## GEM geometry customization
#mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v4/gemf.xml')
#process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v4/gemf.xml')
#process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v5/gemf.xml')
#
#mynum = process.XMLIdealGeometryESSource.geomXMLFiles.index('Geometry/MuonCommonData/data/v2/muonGemNumbering.xml')
#process.XMLIdealGeometryESSource.geomXMLFiles.remove('Geometry/MuonCommonData/data/v2/muonGemNumbering.xml')
#process.XMLIdealGeometryESSource.geomXMLFiles.insert(mynum,'Geometry/MuonCommonData/data/v5/muonGemNumbering.xml')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                                      
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_40_4_vgy.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_41_1_RcW.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_4_1_bxq.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_42_1_fgx.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_43_3_kKG.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_44_1_l0D.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_45_1_sft.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_46_1_zb8.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_47_1_Mk5.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_48_1_6kV.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_49_1_7Gz.root',
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_50_1_RQM.root',
                                     
                                      #####
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_100_1_Bjh.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_101_1_wIH.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_10_1_dzM.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_102_1_knh.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_103_1_pH0.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_104_1_gdj.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_105_1_R17.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_106_1_aqJ.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_107_1_cLZ.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_108_1_RLB.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_109_1_4qK.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_110_1_lzT.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_111_1_aRT.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_11_1_auk.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_112_1_UJM.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_113_1_Naa.root',
#                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_Fixed2023_LXPLUS_DIGIv5/calabria_SingleMuPt50_GEN-SIM-DIGI-RECO_GEMRecoAndStdRECO_62X_SLHC5_Fixed2023_DigiV5/3eb85ef47d9a24f73264373d8a685a2f/out_reco_std_114_1_Ogb.root',


    )
)

process.FILE = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string('histo.root') )

process.load("Validation.MuonGEMRecHits.MuonGEMRecHits_cfi")
#process.RecHitAnalyzer.EffRootFileName="prova2.root"
process.p = cms.Path(process.gemRecHitsValidation)
#process.outpath = cms.EndPath(process.FILE)
