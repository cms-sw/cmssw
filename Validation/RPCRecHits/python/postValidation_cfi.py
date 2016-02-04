import FWCore.ParameterSet.Config as cms

from DQMServices.ClientConfig.genericClientPSetHelper_cff import *

rpcRecHitPostValidation = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("RPC/RPCRecHitV/SimVsReco",
                                    "RPC/RPCRecHitV/SimVsDTExt",
                                    "RPC/RPCRecHitV/SimVsCSCExt"),
    efficiency = cms.vstring(),
    resolution = cms.vstring(),
    efficiencyProfileSets = cms.untracked.VPSet(
        efficSet("Efficiency/Effic_Wheel", "Barrel SimHit to RecHit matching efficiency;Wheel",
                 "Occupancy/NMatchedRefHit_Wheel", "Occupancy/NRefHit_Wheel"),
        efficSet("Efficiency/Effic_Disk", "Endcap SimHit to RecHit matching efficiency;Disk",
                 "Occupancy/NMatchedRefHit_Disk", "Occupancy/NRefHit_Disk"),
        efficSet("Efficiency/Noise_Wheel", "Barrel un-matched RecHit to SimHit efficiency;Wheel",
                 "Occupancy/NUnMatchedRecHit_Wheel","Occupancy/NRecHit_Wheel"),
        efficSet("Efficiency/Noise_Disk", "Endcap un-matched RecHit to SimHit efficiency;Disk",
                 "Occupancy/NUnMatchedRecHit_Disk", "Occupancy/NRecHit_Disk"),
        efficSet("Efficiency/Lost_Wheel", "Barrel un-matched SimHit to RecHit efficiency;Wheel",
                 "Occupancy/NUnMatchedRefHit_Wheel", "Occupancy/NRefHit_Wheel"),
        efficSet("Efficiency/Lost_Disk", "Endcap un-matched SimHit to RecHit efficiency;Disk",
                 "Occupancy/NUnMatchedRefHit_Disk", "Occupancy/NRefHit_Disk"),
    ),
    resolutionSets = cms.untracked.VPSet(
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_W"),
            titlePrefix = cms.untracked.string("Wheel residual"),
            srcName = cms.untracked.string("Residual/Res2_W")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_D"),
            titlePrefix = cms.untracked.string("Disk residual"),
            srcName = cms.untracked.string("Residual/Res2_D")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_W"),
            titlePrefix = cms.untracked.string("Wheel pull"),
            srcName = cms.untracked.string("Residual/Pull2_W")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_D"),
            titlePrefix = cms.untracked.string("Disk pull"),
            srcName = cms.untracked.string("Residual/Pull2_D")
        ),
    ),
    outputFileName = cms.untracked.string("")
)

rpcPointVsRecHitPostValidation = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("RPC/RPCRecHitV/DTVsReco",
                                    "RPC/RPCRecHitV/CSCVsReco"),
#                                    "RPC/RPCRecHitV/TrackVsReco"),
    efficiency = cms.vstring(),
    resolution = cms.vstring(),
    efficiencyProfileSets = cms.untracked.VPSet(
        efficSet("Efficiency/Effic_Wheel", "Barrel RPCPoint to RecHit matching efficiency;Wheel",
                 "Occupancy/NMatchedRefHit_Wheel", "Occupancy/NRefHit_Wheel"),
        efficSet("Efficiency/Effic_Disk", "Endcap RPCPoint to RecHit matching efficiency;Disk",
                 "Occupancy/NMatchedRefHit_Disk", "Occupancy/NRefHit_Disk"),
        efficSet("Efficiency/Noise_Wheel", "Barrel un-matched RecHit to RPCPoint efficiency;Wheel",
                 "Occupancy/NUnMatchedRecHit_Wheel","Occupancy/NRecHit_Wheel"),
        efficSet("Efficiency/Noise_Disk", "Endcap un-matched RecHit to RPCPoint efficiency;Disk",
                 "Occupancy/NUnMatchedRecHit_Disk", "Occupancy/NRecHit_Disk"),
        efficSet("Efficiency/Lost_Wheel", "Barrel un-matched RPCPoint to RecHit efficiency;Wheel",
                 "Occupancy/NUnMatchedRefHit_Wheel", "Occupancy/NRefHit_Wheel"),
        efficSet("Efficiency/Lost_Disk", "Endcap un-matched RPCPoint to RecHit efficiency;Disk",
                 "Occupancy/NUnMatchedRefHit_Disk", "Occupancy/NRefHit_Disk"),
    ),
    resolutionSets = cms.untracked.VPSet(
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_W"),
            titlePrefix = cms.untracked.string("Wheel residual"),
            srcName = cms.untracked.string("Residual/Res2_W")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_D"),
            titlePrefix = cms.untracked.string("Disk residual"),
            srcName = cms.untracked.string("Residual/Res2_D")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_W"),
            titlePrefix = cms.untracked.string("Wheel pull"),
            srcName = cms.untracked.string("Residual/Pull2_W")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_D"),
            titlePrefix = cms.untracked.string("Disk pull"),
            srcName = cms.untracked.string("Residual/Pull2_D")
        ),
    ),
    outputFileName = cms.untracked.string("")
)

rpcRecHitPostValidation_step = cms.Sequence(rpcRecHitPostValidation)
rpcPointVsRecHitPostValidation_step = cms.Sequence(rpcPointVsRecHitPostValidation)
