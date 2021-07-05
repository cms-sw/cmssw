import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("DumpDTRaw",Run3)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    "file:/eos/cms/store/data/Commissioning2021/Cosmics/RAW/v1/000/342/218/00000/fdaf9009-dfd8-4774-9246-556088e65e9b.root",
    "file:/eos/cms/store/data/Commissioning2021/Cosmics/RAW/v1/000/342/094/00000/7e88d2e8-6632-40f0-a1ca-4350adf60182.root"
    ),
                            skipEvents = cms.untracked.uint32(0) )

# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
)        
process.load("CondCore.DBCommon.CondDBSetup_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = "auto:run3_hlt_GRun")
 
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineTestLegacy"),
                                                refreshTime = cms.uint64(1)

                                            ),
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
                                                tag = cms.string('BeamSpotOnlineTestHLT'),
                                                refreshTime = cms.uint64(1)
                                               )

                                ),
                                        #connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')
                                        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
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

process.scalersRawToDigi = cms.EDProducer('ScalersRawToDigi')

process.load("RecoVertex.BeamSpotProducer.BeamSpotOnline_cff")
#process.onlineBeamSpotProducer.useTransientRecord = cms.bool(False)

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( 'onlineBeamSpotwithDB3.root' ),
                                outputCommands = cms.untracked.vstring(
    "keep *_*_*_DumpDTRaw"
    )
                                )

process.dtRawDump = cms.Path(process.scalersRawToDigi + process.onlineBeamSpot )
#process.dtRawDump = cms.Path( process.onlineBeamSpot )

process.e = cms.EndPath( process.out )
