import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)

# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
# emulation
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")
process.load("L1Trigger/RPCTrigger/rpcTriggerDigis_cff")
process.rpcTriggerDigis.label = cms.string('simMuonRPCDigis')


# should I test globalTag?
#process.load("L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi")
#process.rpcconf.filedir = cms.untracked.string('L1Trigger/RPCTrigger/data/D30/')

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
