import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
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
process.GlobalTag.globaltag = "MC_38Y_V9::All"

process.load("Validation.MuonCSCDigis.cscDigiValidation_cfi")

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_8_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0005/62065D40-3D95-DF11-83FA-002618943976.root',
       '/store/relval/CMSSW_3_8_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0004/F0D135F1-0995-DF11-9B08-0018F3D096A2.root',
       '/store/relval/CMSSW_3_8_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0004/CEC41BA0-0895-DF11-B5E9-0018F3D096DA.root',
       '/store/relval/CMSSW_3_8_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0004/521C57A2-0895-DF11-9E35-001BFCDBD160.root' )
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

