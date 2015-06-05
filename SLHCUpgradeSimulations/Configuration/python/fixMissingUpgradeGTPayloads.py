import FWCore.ParameterSet.Config as cms

def fixRPCConditions(process):
    process.simMuonRPCDigis.digiModel = cms.string('RPCSimAverageNoiseEffCls')
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("RPCStripNoisesRcd"),
                 tag = cms.string("RPC_testCondition_192Strips_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_RPC")
                 ),
        cms.PSet(record = cms.string("RPCClusterSizeRcd"),
                 tag = cms.string("RPCClusterSize_PhaseII_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_36X_RPC")
                 )
        )
    )
    return process

def fixDTAlignmentConditions(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"),
                     tag = cms.string("MuonDTAPEObjectsExtended_v0_mc"),
                     connect = cms.string("frontier://FrontierProd/CMS_COND_ALIGN_000")
                 )
            )
    ),
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("DTRecoUncertaintiesRcd"),
                     tag = cms.string("DTRecoUncertainties_True_v0"),
                     connect = cms.string("frontier://FrontierProd/CMS_COND_DT_000")
                 )
            )
    ),
    return process

def fixCSCAlignmentConditions(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"),
                     tag = cms.string("MuonCSCAPEObjectsExtended_v0_mc"),
                     connect = cms.string("frontier://FrontierProd/CMS_COND_ALIGN_000")
                 )
            )
    ),
    return process
