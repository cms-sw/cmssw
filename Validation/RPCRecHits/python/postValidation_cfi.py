import FWCore.ParameterSet.Config as cms

from DQMServices.ClientConfig.genericClientPSetHelper_cff import *

rpcRecHitSimRecoClient = cms.EDAnalyzer("RPCRecHitValidClient",
    subDir = cms.string("RPC/RPCRecHitV/SimVsReco"),
)

rpcRecHitPostValidation = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("RPC/RPCRecHitV/SimVsReco",),
    #subDirs = cms.untracked.vstring("RPC/RPCRecHitV/SimVsReco",
    #                                "RPC/RPCRecHitV/SimVsDTExt",
    #                                "RPC/RPCRecHitV/SimVsCSCExt"),
    efficiency = cms.vstring(),
    resolution = cms.vstring(),
    efficiencyProfileSets = cms.untracked.VPSet(
        efficSet("Efficiency/Effic_wheel", "Barrel SimHit to RecHit matching efficiency;Wheel",
                 "Occupancy/MatchBarrelOccupancy_wheel", "Occupancy/RefHitBarrelOccupancy_wheel"),
        efficSet("Efficiency/Effic_station", "Barrel SimHit to RecHit matching efficiency;Station",
                 "Occupancy/MatchBarrelOccupancy_station", "Occupancy/RefHitBarrelOccupancy_station"),
        efficSet("Efficiency/Effic_disk", "Endcap SimHit to RecHit matching efficiency;Disk",
                 "Occupancy/MatchEndcapOccupancy_disk", "Occupancy/RefHitEndcapOccupancy_disk"),
        efficSet("Efficiency/Noise_wheel", "Barrel un-matched RecHit to SimHit efficiency;Wheel",
                 "Occupancy/UmBarrelOccupancy_wheel","Occupancy/RecHitBarrelOccupancy_wheel"),
        efficSet("Efficiency/Noise_station", "Barrel un-matched RecHit to SimHit efficiency;Station",
                 "Occupancy/UmBarrelOccupancy_station","Occupancy/RecHitBarrelOccupancy_station"),
        efficSet("Efficiency/Noise_disk", "Endcap un-matched RecHit to SimHit efficiency;Disk",
                 "Occupancy/UmEndcapOccupancy_disk", "Occupancy/RecHitEndcapOccupancy_disk"),
        #efficSet("Efficiency/Lost_wheel", "Barrel un-matched SimHit to RecHit efficiency;Wheel",
        #         "Occupancy/NUnMatchedRefHit_wheel", "Occupancy/RefHitBarrelOccupancy_wheel"),
        #efficSet("Efficiency/Lost_disk", "Endcap un-matched SimHit to RecHit efficiency;Disk",
        #         "Occupancy/NUnMatchedRefHit_disk", "Occupancy/RefHitEndcapOccupancy_disk"),
    ),
    resolutionSets = cms.untracked.VPSet(
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_wheel"),
            titlePrefix = cms.untracked.string("Wheel residual"),
            srcName = cms.untracked.string("Residual/Res_wheel_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_station"),
            titlePrefix = cms.untracked.string("Station residual"),
            srcName = cms.untracked.string("Residual/Res_station_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_disk"),
            titlePrefix = cms.untracked.string("Disk residual"),
            srcName = cms.untracked.string("Residual/Res_disk_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_wheel"),
            titlePrefix = cms.untracked.string("Wheel pull"),
            srcName = cms.untracked.string("Residual/Pull_wheel_pull")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_station"),
            titlePrefix = cms.untracked.string("Station pull"),
            srcName = cms.untracked.string("Residual/Pull_station_pull")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_disk"),
            titlePrefix = cms.untracked.string("Disk pull"),
            srcName = cms.untracked.string("Residual/Pull_disk_pull")
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
        efficSet("Efficiency/Effic_wheel", "Barrel RPCPoint to RecHit matching efficiency;Wheel",
                 "Occupancy/MatchBarrelOccupancy_wheel", "Occupancy/RefHitBarrelOccupancy_wheel"),
        efficSet("Efficiency/Effic_station", "Barrel RPCPoint to RecHit matching efficiency;Station",
                 "Occupancy/MatchBarrelOccupancy_station", "Occupancy/RefHitBarrelOccupancy_station"),
        efficSet("Efficiency/Effic_disk", "Endcap RPCPoint to RecHit matching efficiency;Disk",
                 "Occupancy/MatchEndcapOccupancy_disk", "Occupancy/RefHitEndcapOccupancy_disk"),
        efficSet("Efficiency/Noise_wheel", "Barrel un-matched RecHit to RPCPoint efficiency;Wheel",
                 "Occupancy/UmBarrelOccupancy_wheel","Occupancy/RecHitBarrelOccupancy_wheel"),
        efficSet("Efficiency/Noise_station", "Barrel un-matched RecHit to RPCPoint efficiency;Station",
                 "Occupancy/UmBarrelOccupancy_station","Occupancy/RecHitBarrelOccupancy_station"),
        efficSet("Efficiency/Noise_disk", "Endcap un-matched RecHit to RPCPoint efficiency;Disk",
                 "Occupancy/UmEndcapOccupancy_disk", "Occupancy/RecHitEndcapOccupancy_disk"),
        #efficSet("Efficiency/Lost_wheel", "Barrel un-matched RPCPoint to RecHit efficiency;Wheel",
        #         "Occupancy/NUnMatchedRefHit_wheel", "Occupancy/RefHitBarrelOccupancy_wheel"),
        #efficSet("Efficiency/Lost_disk", "Endcap un-matched RPCPoint to RecHit efficiency;Disk",
        #         "Occupancy/NUnMatchedRefHit_disk", "Occupancy/RefHitEndcapOccupancy_disk"),
    ),
    resolutionSets = cms.untracked.VPSet(
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_wheel"),
            titlePrefix = cms.untracked.string("Wheel residual"),
            srcName = cms.untracked.string("Residual/Res_wheel_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_station"),
            titlePrefix = cms.untracked.string("Station residual"),
            srcName = cms.untracked.string("Residual/Res_station_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Res_disk"),
            titlePrefix = cms.untracked.string("Disk residual"),
            srcName = cms.untracked.string("Residual/Res_disk_res")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_wheel"),
            titlePrefix = cms.untracked.string("Wheel pull"),
            srcName = cms.untracked.string("Residual/Pull_wheel_pull")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_station"),
            titlePrefix = cms.untracked.string("Station pull"),
            srcName = cms.untracked.string("Residual/Pull_station_pull")
        ),
        cms.PSet(
            namePrefix = cms.untracked.string("Resolution/Pull_disk"),
            titlePrefix = cms.untracked.string("Disk pull"),
            srcName = cms.untracked.string("Residual/Pull_disk_pull")
        ),
    ),
    outputFileName = cms.untracked.string("")
)

rpcRecHitPostValidation_step = cms.Sequence(rpcRecHitPostValidation+rpcRecHitSimRecoClient)
rpcPointVsRecHitPostValidation_step = cms.Sequence(rpcPointVsRecHitPostValidation)
