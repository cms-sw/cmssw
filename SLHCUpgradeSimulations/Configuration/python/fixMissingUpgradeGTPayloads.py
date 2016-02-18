import FWCore.ParameterSet.Config as cms

def fixRPCConditions(process):
    if hasattr(process,'simMuonRPCDigis'):
    	process.simMuonRPCDigis.digiModel = cms.string('RPCSimAverageNoiseEffCls')
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("RPCStripNoisesRcd"),
                 tag = cms.string("RPC_testCondition_192Strips_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        cms.PSet(record = cms.string("RPCClusterSizeRcd"),
                 tag = cms.string("RPCClusterSize_PhaseII_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
    )
    return process


