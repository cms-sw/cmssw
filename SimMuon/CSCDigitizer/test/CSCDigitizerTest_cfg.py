import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
#untracked PSet maxEvents = {untracked int32 input = 100}
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#include "SimMuon/CSCDigitizer/data/muonCSCDbConditions.cfi"
#replace muonCSCDigis.stripConditions = "Database"
#replace muonCSCDigis.strips.ampGainSigma = 0.
#replace muonCSCDigis.strips.peakTimeSigma = 0.
#replace muonCSCDigis.strips.doNoise = false
#replace muonCSCDigis.wires.doNoise = false
#replace muonCSCDigis.strips.doCrosstalk = false
process.load("CalibMuon.Configuration.CSC_FakeDBConditions_cff")

#   include "CalibMuon/Configuration/data/CSC_FrontierConditions.cff"
#   replace cscConditions.toGet =  {
#        { string record = "CSCDBGainsRcd"
#          string tag = "CSCDBGains_ideal"},
#        {string record = "CSCNoiseMatrixRcd"
#          string tag = "CSCNoiseMatrix_ideal"},
#        {string record = "CSCcrosstalkRcd"
#          string tag = "CSCCrosstalk_ideal"},
#        {string record = "CSCPedestalsRcd"
#         string tag = "CSCPedestals_ideal"}
#    }
process.load("Validation.MuonCSCDigis.cscDigiValidation_cfi")

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:simevent.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simMuonCSCDigis = cms.untracked.uint32(468)
    )
)

process.DQMStore = cms.Service("DQMStore")

process.dump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    empt = cms.InputTag(""),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('cscdigis.root')
)

process.p1 = cms.Path(process.mix*process.simMuonCSCDigis*process.cscDigiValidation)
process.ep = cms.EndPath(process.o1)

