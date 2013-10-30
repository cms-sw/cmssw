import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMQualityFromDigi")
process.load("DQMServices.Core.DQM_cfg")

#process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

#process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load('Configuration.Geometry.GeometrySimDB_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')

#process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
#process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'DES19_61_V5::All'
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


readFiles = cms.untracked.vstring()

process.source = cms.Source ("PoolSource",fileNames = readFiles)


readFiles.extend( [

#'root://eoscms//eos/cms/store/user/mileva/gemTest/minBias_digiL1digi2raw/minBias_digi_l1_digi2raw_SvenInstr.root'
#'root://eoscms//eos/cms/store/user/mileva/gemTest/singleMuPt40_digiL1digi2raw/singleMuPt40Fwr_digi_l1_digi2raw_SvenInstr.root'
'file:outputDigi.root'
     ] );


process.dqmSource = cms.EDAnalyzer("DQMSourceExample",
    monitorName = cms.untracked.string('YourSubsystemName'),
    prescaleEvt = cms.untracked.int32(-1)
)

process.qTester = cms.EDFilter("QualityTester")

#load module defaults
process.load("Validation.MuonGEMDigis.validationMuonGEMDigis_cfi")


#Overwriting default values
process.validationMuonGEMDigis.outputFile = cms.untracked.string('gemDigiValidPlots.root')

process.p = cms.Path(process.dqmSource+process.validationMuonGEMDigis)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''


