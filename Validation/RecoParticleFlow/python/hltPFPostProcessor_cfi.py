import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltPFClusterPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/ParticleFlow/PFClusterValidation"),
    efficiency = cms.vstring(
        "Eff_vs_EnergyEta 'Efficiency vs Energy-#eta' SimClustersMatchedRecoClustersEnergy_Eta SimClustersEnergy_Eta",
        "Eff_vs_EnergyPhi 'Efficiency vs Energy-#phi' SimClustersMatchedRecoClustersEnergy_Phi SimClustersEnergy_Phi",
        "Eff_vs_EnergyMult 'Efficiency vs Energy-Mult' SimClustersMatchedRecoClustersEnergy_Mult SimClustersEnergy_Mult",
        "Eff_vs_PtEta 'Efficiency vs p_{T}-#eta' SimClustersMatchedRecoClustersPt_Eta SimClustersPt_Eta",
        "Eff_vs_PtPhi 'Efficiency vs p_{T}-#phi' SimClustersMatchedRecoClustersPt_Phi SimClustersPt_Phi",
        "Eff_vs_PtMult 'Efficiency vs p_{T}-Mult' SimClustersMatchedRecoClustersPt_Mult SimClustersPt_Mult",
        "Eff_vs_MultEta 'Efficiency vs Mult-#eta' SimClustersMatchedRecoClustersMult_Eta SimClustersMult_Eta",
        "Eff_vs_MultPhi 'Efficiency vs Mult-#phi' SimClustersMatchedRecoClustersMult_Phi SimClustersMult_Phi",
        "Fake_vs_EnergyEta 'Fake Rate vs Energy-#eta' RecoClustersMatchedSimClustersEnergy_Eta RecoClustersPt_Eta fake",
        "Fake_vs_EnergyPhi 'Fake Rate vs Energy-#phi' RecoClustersMatchedSimClustersEnergy_Phi RecoClustersPt_Phi fake",
        "Fake_vs_EnergyMult 'Fake Rate vs Energy-Mult' RecoClustersMatchedSimClustersEnergy_Mult RecoClustersPt_Mult fake",
        "Fake_vs_PtEta 'Fake Rate vs p_{T}-#eta' RecoClustersMatchedSimClustersPt_Eta RecoClustersPt_Eta fake",
        "Fake_vs_PtPhi 'Fake Rate vs p_{T}-#phi' RecoClustersMatchedSimClustersPt_Phi RecoClustersPt_Phi fake",
        "Fake_vs_PtMult 'Fake Rate vs p_{T}-Mult' RecoClustersMatchedSimClustersPt_Mult RecoClustersPt_Mult fake",
        "Fake_vs_MultEta 'Fake Rate vs Mult-#eta' RecoClustersMatchedSimClustersMult_Eta RecoClustersMult_Eta fake",
        "Fake_vs_MultPhi 'Fake Rate vs Mult-#phi' RecoClustersMatchedSimClustersMult_Phi RecoClustersMult_Phi fake",
    ),
    efficiencyProfile = cms.untracked.vstring( # for smoother rebinning
        # Efficiency
        "Eff_vs_Energy 'Efficiency vs Energy' SimClustersMatchedRecoClustersEnergy SimClustersEnergy ",
        "Eff_vs_Pt 'Efficiency vs p_{T}' SimClustersMatchedRecoClustersPt SimClustersPt ",
        "Eff_vs_Eta 'Efficiency vs #eta' SimClustersMatchedRecoClustersEta SimClustersEta ",
        "Eff_vs_Phi 'Efficiency vs #phi' SimClustersMatchedRecoClustersPhi SimClustersPhi ",
        "Eff_vs_Mult 'Efficiency vs Multiplicity' SimClustersMatchedRecoClustersMult SimClustersMult ",
        # Fake rate
        "Fake_vs_Energy 'Fake Rate vs Energy' RecoClustersMatchedSimClustersEnergy RecoClustersEnergy ",
        "Fake_vs_Pt 'Fake Rate vs p_{T}' RecoClustersMatchedSimClustersPt RecoClustersPt ",
        "Fake_vs_Eta 'Fake Rate vs #eta' RecoClustersMatchedSimClustersEta RecoClustersEta ",
        "Fake_vs_Phi 'Fake Rate vs #phi' RecoClustersMatchedSimClustersPhi RecoClustersPhi ",
        "Fake_vs_Mult 'Fake Rate vs Multiplicity' RecoClustersMatchedSimClustersMult RecoClustersMult ",
        # Duplicate rate
        "Dup_vs_Energy 'Dup Rate vs Energy' RecoClustersMultiMatchedSimClustersEnergy RecoClustersEnergy ",
        "Dup_vs_Pt 'Dup Rate vs p_{T}' RecoClustersMultiMatchedSimClustersPt RecoClustersPt ",
        "Dup_vs_Eta 'Dup Rate vs #eta' RecoClustersMultiMatchedSimClustersEta RecoClustersEta ",
        "Dup_vs_Phi 'Dup Rate vs #phi' RecoClustersMultiMatchedSimClustersPhi RecoClustersPhi ",
        "Dup_vs_Mult 'Dup Rate vs Mult' RecoClustersMultiMatchedSimClustersMult RecoClustersMult ",
        # Merge rate
        "Merge_vs_Energy 'Merge Rate vs Energy' SimClustersMultiMatchedRecoClustersEnergy SimClustersEnergy ",
        "Merge_vs_Pt 'Merge Rate vs p_{T}' SimClustersMultiMatchedRecoClustersPt SimClustersPt ",
        "Merge_vs_Eta 'Merge Rate vs #eta' SimClustersMultiMatchedRecoClustersEta SimClustersEta ",
        "Merge_vs_Phi 'Merge Rate vs #phi' SimClustersMultiMatchedRecoClustersPhi SimClustersPhi ",
        "Merge_vs_Mult 'Merge Rate vs Multiplicity' SimClustersMultiMatchedRecoClustersMult SimClustersMult ",
    ),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(2), 
    outputFileName = cms.untracked.string("")
)
