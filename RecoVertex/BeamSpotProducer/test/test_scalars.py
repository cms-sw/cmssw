import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDTRaw")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/BeamCommissioning09/RandomTriggers/RAW/v1/000/123/576/4AD8F322-B4E1-DE11-BFF2-0030487A322E.root'
    ),
                            skipEvents = cms.untracked.uint32(0) )

# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(15)
)        


process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
                                    )

process.scalers = cms.EDProducer('ScalersRawToDigi')
process.onlineBeamSpot = cms.EDProducer('BeamSpotOnlineProducer',
                              label = cms.InputTag('scalers')
                              )

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( 'onlineBeamSpot.root' ),
                                outputCommands = cms.untracked.vstring(
    "keep *"
    )
                                )

process.dtRawDump = cms.Path(process.scalers + process.onlineBeamSpot )

process.e = cms.EndPath( process.out )
