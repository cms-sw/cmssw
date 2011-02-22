import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
#untracked PSet maxEvents = {untracked int32 input = 100}
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#include "SimMuon/CSCDigitizer/data/muonCSCDbConditions.cfi"
#replace muonCSCDigis.stripConditions = "Database"
#replace muonCSCDigis.strips.ampGainSigma = 0.
#replace muonCSCDigis.strips.peakTimeSigma = 0.
#replace muonCSCDigis.strips.doNoise = false
#replace muonCSCDigis.wires.doNoise = false
#replace muonCSCDigis.strips.doCrosstalk = false
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_V9::All"


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
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/BA6DA407-A985-DD11-A0D8-000423D9A2AE.root',
       '/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/CAADA54A-AB85-DD11-95D6-000423D98F98.root',
       '/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/52BAB7DF-0487-DD11-95A3-000423D9989E.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simMuonCSCDigis = cms.untracked.uint32(468)
    )
)

process.DQMStore = cms.Service("DQMStore")

process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")

#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('cscdigis.root')
#)

process.p1 = cms.Path(process.mix*process.simMuonCSCDigis*process.cscSimDigiDump)
#process.ep = cms.EndPath(process.o1)
#

