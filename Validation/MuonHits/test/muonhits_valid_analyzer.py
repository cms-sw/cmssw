import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonHitsValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry 
#process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Alignment.CommonAlignmentProducer.test.GlobalPositionRcd_read_cfg.py") 
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")
process.load("Configuration.StandardSequences.FakeConditions_cff")


# DQM services
process.load("DQMServices.Core.DQM_cfg")

#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# run simulation, with EcalHits Validation specific watcher 
#process.load("SimG4Core.Application.g4SimHits_cfi")

# DT Muon hits validation sequence
process.load("Validation.MuonHits.muonHitsValidation_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:Pion_Pt60GeV_all.root')
      fileNames = cms.untracked.vstring(
## CMSSW_210_pre9 relval files
# '/store/relval/2008/7/21/RelVal-RelValSingleMuPt100-1216579481-IDEAL_V5-2nd/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/940DEAA6-6E57-DD11-A383-000423D98800.root',
#         '/store/relval/2008/7/21/RelVal-RelValSingleMuPt100-1216579481-IDEAL_V5-2nd/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/E0124BEF-6E57-DD11-9696-000423D98834.root',
#        '/store/relval/2008/7/21/RelVal-RelValSingleMuPt100-1216579481-IDEAL_V5-2nd/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/F632DE6B-6F57-DD11-8A2C-001617C3B77C.root')
## CMSSW_200_pre4 relval 
#'/store/relval/2008/3/12/RelVal-RelValSingleMuPlusPt100-1205356055/0000/44DB3F3D-CBF0-DC11-92F5-000423D6C8E6.root',
#  '/store/relval/2008/3/12/RelVal-RelValSingleMuPlusPt100-1205356055/0000/FEFBEE7D-C9F0-DC11-A5DC-000423D98844.root'
## CMSSW_214 relval  
#       '/store/relval/CMSSW_2_1_4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/400B03E1-2A6C-DD11-A78C-001617E30D4A.root')
        '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/2E5CADBF-7B6C-DD11-9888-0019DB29C614.root')
)

process.USER = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('Muon_HitsValidation.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876)
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

#process.simhits = cms.Sequence(process.g4SimHits*process.ecalSimHitsValidationSequence)
#process.validation = cms.Sequence(process.muonHitsValidation)
process.validation = cms.Sequence(process.validSimHit)
process.p1 = cms.Path(process.validation)
process.p4 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.USER)
#process.schedule = cms.Schedule(process.p1,process.p4,process.outpath)
process.schedule = cms.Schedule(process.p1)

process.DQM.collectorHost = ''
#process.g4SimHits.Generator.HepMCProductLabel = 'source'
#process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
#    instanceLabel = cms.untracked.string('EcalValidInfo'),
#    type = cms.string('EcalSimHitsValidProducer'),
#    verbose = cms.untracked.bool(False)
#))




