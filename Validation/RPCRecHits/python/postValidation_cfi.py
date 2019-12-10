import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

def efficSet(nameIn, titleIn, numeratorIn, denominatorIn, typeIn="eff"):
    pset = cms.PSet(name=cms.untracked.string(nameIn),
                    title=cms.untracked.string(titleIn), 
                    numerator=cms.untracked.string(numeratorIn), 
                    denominator=cms.untracked.string(denominatorIn),
                    type=cms.untracked.string(typeIn))
    return pset

rpcRecHitSimRecoClient = DQMEDHarvester("RPCRecHitValidClient",
    subDir = cms.string("RPC/RPCRecHitV/SimVsReco"),
)

rpcRecHitPostValidation = DQMEDHarvester("DQMGenericClient",
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

rpcPointVsRecHitPostValidation = DQMEDHarvester("DQMGenericClient",
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
