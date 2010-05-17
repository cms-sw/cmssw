import FWCore.ParameterSet.Config as cms

rpcRecHitPostValidation = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("RPC/RPCRecHitV/SimVsReco"),
    efficiency = cms.vstring(
        "Effic_Eta 'Efficiency in #eta;Pseudorapidity #eta' MatchedRecHitEta SimHitEta",
        "Effic_Wheel 'Barrel SimHit to RecHit matching efficiency;Wheel' NMatchedRecHit_Wheel NSimHit_Wheel",
        "Effic_Disk 'Endcap SimHit to RecHit matching efficiency;Disk' NMatchedRecHit_Disk NSimHit_Disk",
        "NoiseRate_Eta 'Noise rate in #eta;Pseudorapidity #eta' NoisyHitEta RecHitEta",
        "NoiseRate_Wheel 'Barrel un-matched RecHit to SimHit rate;Wheel' NNoisyHit_Wheel NRecHit_Wheel",
        "NoiseRate_Disk 'Endcap un-matched RecHit to SimHit rate;Disk' NNoisyHit_Disk NRecHit_Disk",
        "LostRate_Wheel 'Barrel un-matched SimHit to RecHit rate;Wheel' NLostHit_Wheel NSimHit_Wheel",
        "LostRate_Disk 'Endcap un-matched SimHit to RecHit rate;Disk' NLostHit_Disk NSimHit_Disk"
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

rpcPointVsRecHitPostValidation = cms.EDAnalyzer("DQMGenericClient",
    subDirs = cms.untracked.vstring("RPC/RPCRecHitV/DTVsReco",
                                    "RPC/RPCRecHitV/CSCVsReco",
                                    "RPC/RPCRecHitV/TrackVsReco"),
    efficiency = cms.vstring(
        "Effic_Eta 'Efficiency in #eta;Pseudorapidity #eta' MatchedRecHitEta RefHitEta",
        "Effic_Wheel 'Barrel RPCPoint to RecHit matching efficiency;Wheel' NMatchedRecHit_Wheel NRefHit_Wheel",
        "Effic_Disk 'Endcap RPCPoint to RecHit matching efficiency;Disk' NMatchedRecHit_Disk NRefHit_Disk",
        "NoiseRate_Eta 'Noise rate in #eta;Pseudorapidity #eta' NoisyHitEta ../SimVsReco/RecHitEta",
        "NoiseRate_Wheel 'Barrel un-matched RecHit to RPCPoint rate;Wheel' NNoisyHit_Wheel NRecHit_Wheel",
        "NoiseRate_Disk 'Endcap un-matched RecHit to RPCPoint rate;Disk' NNoisyHit_Disk NRecHit_Disk",
        "LostRate_Wheel 'Barrel un-matched RPCPoint to RecHit rate;Wheel' NLostHit_Wheel NRefHit_Wheel",
        "LostRate_Disk 'Endcap un-matched RPCPoint to RecHit rate;Disk' NLostHit_Disk NRefHit_Disk"
    ),
    resolution = cms.vstring(""),
    outputFileName = cms.untracked.string("")
)

rpcRecHitPostValidation_step = cms.Sequence(rpcRecHitPostValidation)
rpcPointVsRecHitPostValidation_step = cms.Sequence(rpcPointVsRecHitPostValidation)
