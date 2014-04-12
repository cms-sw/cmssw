import FWCore.ParameterSet.Config as cms

process = cms.Process('CSCNoiseMatrixTest')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
# Other statements
#process.GlobalTag.globaltag = 'IDEAL_V11::All'
#process.GlobalTag.globaltag = 'IDEAL_V9::All'
process.load("CalibMuon.Configuration.getCSCConditions_frontier_cff")
process.cscConditions.connect='oracle://cms_orcoff_prep/CMS_COND_CSC'
#process.cscConditions.connect = 'sqlite_file:DBNoiseMatrix.db'
process.cscConditions.toGet = cms.VPSet(
        cms.PSet(record = cms.string('CSCDBGainsRcd'),
                 tag = cms.string('CSCDBGains_ME42_offline')),
        cms.PSet(record = cms.string('CSCDBNoiseMatrixRcd'),
                 tag = cms.string('CSCDBNoiseMatrix_ME42_March2009')),
        cms.PSet(record = cms.string('CSCDBCrosstalkRcd'),
                 tag = cms.string('CSCDBCrosstalk_ME42_offline')),
        cms.PSet(record = cms.string('CSCDBPedestalsRcd'),
                 tag = cms.string('CSCDBPedestals_ME42_offline'))
)

process.es_prefer_cscConditions = cms.ESPrefer("PoolDBESSource","cscConditions")
#process.es_prefer_cscBadChambers = cms.ESPrefer("PoolDBESSource","cscBadChambers")

process.cscConditions.DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
#        authenticationPath = cms.untracked.string('/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_2_2_3/src/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
)
process.CSCGeometryESModule.applyAlignment = False

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")
process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.cscNoiseTest = cms.EDAnalyzer("CSCNoiseMatrixTest",
      readBadChannels = cms.bool(False),
      readBadChambers = cms.bool(True),
      doCrosstalk = cms.bool(True),
      gainsConstant = cms.double(0.27),
      capacativeCrosstalk = cms.double(35.0),
      resistiveCrosstalkScaling = cms.double(1.8),
      doCorrelatedNoise = cms.bool(True))

process.RandomNumberGeneratorService.cscNoiseTest = cms.PSet(
                   engineName = cms.untracked.string('HepJamesRandom'),
                   initialSeed = cms.untracked.uint32(1234)
)


process.path = cms.Path(process.cscNoiseTest)
