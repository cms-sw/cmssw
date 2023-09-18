import FWCore.ParameterSet.Config as cms

process = cms.Process("DTValidationFromRAW")

## General CMS
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "CRUZET4_V3P::All"

process.load("Configuration.StandardSequences.MagneticField_cff")

# process.calibDB = cms.ESSource("PoolDBESSource",
#     process.CondDBSetup,
#     authenticationMethod = cms.untracked.uint32(0),
#     toGet = cms.VPSet(cms.PSet(
#         # VDrift
#         #string record = "DTMtimeRcd"
#         #string tag ="vDrift"
#         # TZero
#         #string record = "DTT0Rcd" 
#         #string tag = "t0"
#         #string tag = "t0_GRUMM"
#         # TTrig
#         record = cms.string('DTTtrigRcd'),
#         tag = cms.string('ttrig')
#     )),
#     connect = cms.string('sqlite_file:/afs/cern.ch/user/p/pellicci/scratch0/DPG/LocalReco/CMSSW_3_1_X_2009-02-25-0700/src/CalibMuon/DTCalibration/test/ttrig.db')
# )
#process.prefer = cms.ESPrefer("PoolDBESSource","calibDB")

#Geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

## DT unpacker
process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")
process.muonDTDigis.inputLabel = 'rawDataCollector'

## DT local Reco
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")
process.dt1DRecHits.dtDigiLabel  = cms.InputTag("muonDTDigis")

# Validation RecHits
process.load("Validation.DTRecHits.DTRecHitQuality_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.options = cms.untracked.PSet(
    #TryToContinue = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/746C9E4E-D932-DE11-B1E6-001617DBCF90.root',
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/5C7FE942-1733-DE11-880D-001617C3B77C.root',
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/32F159D3-D832-DE11-9A86-000423D98A44.root'
    )
                            )

process.muonLocalReco = cms.Sequence(process.dtlocalreco_with_2DSegments)

process.analysis = cms.Sequence(process.dtLocalRecoValidation)

process.p = cms.Path(process.muonDTDigis *
                     process.muonLocalReco *
                     process.analysis)

