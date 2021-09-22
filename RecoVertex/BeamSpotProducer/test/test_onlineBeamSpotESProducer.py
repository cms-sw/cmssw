import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("DumpDTRaw",Run3)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    "file:/eos/cms/store/data/Commissioning2021/Cosmics/RAW/v1/000/344/518/00000/00130ffc-3e69-4106-8151-c69dd735ee2e.root",
    "file:/eos/cms/store/data/Commissioning2021/Cosmics/RAW/v1/000/344/518/00000/001d10b6-a19e-4889-ba07-88f8f6b17bc7.root"
    ),
                            skipEvents = cms.untracked.uint32(0) )

# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
)
process.load("CondCore.CondDB.CondDB_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = "auto:run3_hlt_GRun")

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(
    record = cms.string("BeamSpotOnlineLegacyObjectsRcd"),
    refreshTime = cms.uint64(1)
  ),
  cms.PSet(
    record = cms.string("BeamSpotOnlineHLTObjectsRcd"),
    refreshTime = cms.uint64(1)
  )
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cff")
#process.onlineBeamSpotProducer.useTransientRecord = cms.bool(False)

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( 'onlineBeamSpotwithDB3.root' ),
                                outputCommands = cms.untracked.vstring(
    "keep *_*_*_DumpDTRaw"
    )
                                )

process.dtRawDump = cms.Path( process.onlineBeamSpot )

process.e = cms.EndPath( process.out )
