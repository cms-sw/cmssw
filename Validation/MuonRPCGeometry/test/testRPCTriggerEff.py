import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")


#process.load("FWCore.MessageLogger.MessageLogger_cfi")
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


# rpc r2d
#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
#process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt1000/GEN-SIM-RECO/IDEAL_31X_v1/0004/EC465F32-C941-DE11-A41B-001D09F2441B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt1000/GEN-SIM-RECO/IDEAL_31X_v1/0004/42075C89-E641-DE11-9864-001D09F2441B.root' ,
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0004/B286E67D-E641-DE11-96CD-001D09F2527B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0004/36C61875-D141-DE11-AB22-001D09F2447F.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/B89FA4AB-CC41-DE11-8348-000423D98B28.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/58AEC160-E741-DE11-AE53-001D09F2AF1E.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/50F087DB-CB41-DE11-98F2-0030487C5CFA.root' 
    )
)

process.rpceff  = cms.EDFilter("RPCPhiEff",
      #rpcb = cms.InputTag("rpcTriggerDigis:RPCb"),
      #rpcf = cms.InputTag("rpcTriggerDigis:RPCf"),
 #     rpcb = cms.InputTag("simRpcTriggerDigis:RPCb"),
 #     rpcf = cms.InputTag("simRpcTriggerDigis:RPCf"),
 # for RECO
      rpcb = cms.InputTag("gtDigis:RPCb"),
      rpcf = cms.InputTag("gtDigis:RPCf"),
      rpcdigi = cms.InputTag("simMuonRPCDigis"),
      g4 = cms.InputTag("g4SimHits")

)

process.a = cms.Path(process.rpceff)
