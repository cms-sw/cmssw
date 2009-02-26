import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")


#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
#    log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
#    debugModules = cms.untracked.vstring("rpcTriggerDigis"),
    destinations = cms.untracked.vstring('cout')
    #destinations = cms.untracked.vstring('cout/dev/null')
)

# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

# emulation
process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeSource_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfigSource_cfi")
process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cfi")

# one config to rule them all
#process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cff")

# should I test globalTag?
process.load("L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi")
process.rpcconf.filedir = cms.untracked.string('L1Trigger/RPCTrigger/data/D30/')

#process.rpcTriggerDigis.buildOwnLinkSystem = cms.bool(True)
process.rpcTriggerDigis.label = cms.string('simMuonRPCDigis')

# rpc r2d
#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
#process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        "file:/tmp/fruboes/digi_11.root"
    )
)

process.rpceff  = cms.EDFilter("RPCPhiEff",
      rpcb = cms.InputTag("rpcTriggerDigis:RPCb"),
      rpcf = cms.InputTag("rpcTriggerDigis:RPCf"),
      rpcdigi = cms.InputTag("simMuonRPCDigis"),
      g4 = cms.InputTag("g4SimHits")

)

process.a = cms.Path(process.rpcTriggerDigis*process.rpceff)
