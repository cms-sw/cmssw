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


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
                                      
                                      'file:/lustre/cms/store/user/calabria/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/calabria_SingleMuPt200_GEN-SIM-DIGI-RECO_CMSSW_6_2_0_SLHC5_8EtaPar_LXPLUS_DIGIv4_GeomV5/6478f1063444c74e603334fdb6f73345/out_reco_100_2_9Px.root',
    )
)

process.FILE = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string('histo.root') )

process.load("Validation.MuonGEMRecHits.MuonGEMRecHits_cfi")
#process.RecHitAnalyzer.EffRootFileName="prova2.root"
process.p = cms.Path(process.gemRecHitsValidation)
#process.outpath = cms.EndPath(process.FILE)
