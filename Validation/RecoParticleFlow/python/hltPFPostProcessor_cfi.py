import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Validation.RecoParticleFlow.hltPFValidation_cfi import hltPFTesterECAL
_thresholds = [str(x).replace('.', 'p') for x in hltPFTesterECAL.assocScoreThresholds]

hltPFClusterPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/ParticleFlow/PFClusterValidation*"),
    efficiency = cms.vstring(
        *[ item
            for thr in _thresholds
            for name, suf in zip(('', '_Reconstructable'), ('', 'Reconstructable'))
            for item in (
                    f"'Score{thr}/Eff_vs_EnEta{name}' 'Efficiency vs Energy-#eta {suf}' Score{thr}/SimClustersMatchedRecoClustersEn_Eta SimClusters{suf}En_Eta",          
                    f"'Score{thr}/Eff_vs_EnPhi{name}' 'Efficiency vs Energy-#phi {suf}' Score{thr}/SimClustersMatchedRecoClustersEn_Phi SimClusters{suf}En_Phi",          
                    f"'Score{thr}/Eff_vs_EnMult{name}' 'Efficiency vs Energy-Mult {suf}' Score{thr}/SimClustersMatchedRecoClustersEn_Mult SimClusters{suf}En_Mult",       
                    f"'Score{thr}/Eff_vs_EnHitsEta{name}' 'Efficiency vs Hits Energy-#eta {suf}' Score{thr}/SimClustersMatchedRecoClustersEnHits_Eta SimClusters{suf}EnHits_Eta",          
                    f"'Score{thr}/Eff_vs_EnHitsPhi{name}' 'Efficiency vs Hits Energy-#phi {suf}' Score{thr}/SimClustersMatchedRecoClustersEnHits_Phi SimClusters{suf}EnHits_Phi",          
                    f"'Score{thr}/Eff_vs_EnHitsMult{name}' 'Efficiency vs Hits Energy-Mult {suf}' Score{thr}/SimClustersMatchedRecoClustersEnHits_Mult SimClusters{suf}EnHits_Mult",       
                    f"'Score{thr}/Eff_vs_EnFracEta{name}' 'Efficiency vs Energy fraction-#eta {suf}' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Eta SimClusters{suf}EnFrac_Eta",          
                    f"'Score{thr}/Eff_vs_EnFracPhi{name}' 'Efficiency vs Energy fraction-#phi {suf}' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Phi SimClusters{suf}EnFrac_Phi",          
                    f"'Score{thr}/Eff_vs_EnFracMult{name}' 'Efficiency vs Energy fraction-Mult {suf}' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Mult SimClusters{suf}EnFrac_Mult",       
                    f"'Score{thr}/Eff_vs_PtEta{name}' 'Efficiency vs p_{{T}}-#eta {suf}' Score{thr}/SimClustersMatchedRecoClustersPt_Eta SimClusters{suf}Pt_Eta",             
                    f"'Score{thr}/Eff_vs_PtPhi{name}' 'Efficiency vs p_{{T}}-#phi {suf}' Score{thr}/SimClustersMatchedRecoClustersPt_Phi SimClusters{suf}Pt_Phi",             
                    f"'Score{thr}/Eff_vs_PtMult{name}' 'Efficiency vs p_{{T}}-Mult {suf}' Score{thr}/SimClustersMatchedRecoClustersPt_Mult SimClusters{suf}Pt_Mult",          
                    f"'Score{thr}/Eff_vs_MultEta{name}' 'Efficiency vs Mult-#eta {suf}' Score{thr}/SimClustersMatchedRecoClustersMult_Eta SimClusters{suf}Mult_Eta",          
                    f"'Score{thr}/Eff_vs_MultPhi{name}' 'Efficiency vs Mult-#phi {suf}' Score{thr}/SimClustersMatchedRecoClustersMult_Phi SimClusters{suf}Mult_Phi",          
            )
            ],
        *[ item
            for thr in _thresholds
            for item in (    
                    f"'Score{thr}/Fake_vs_EnEta' 'Fake Rate vs Energy-#eta' Score{thr}/RecoClustersMatchedSimClustersEn_Eta RecoClustersPt_Eta fake",    
                    f"'Score{thr}/Fake_vs_EnPhi' 'Fake Rate vs Energy-#phi' Score{thr}/RecoClustersMatchedSimClustersEn_Phi RecoClustersPt_Phi fake",    
                    f"'Score{thr}/Fake_vs_EnMult' 'Fake Rate vs Energy-Mult' Score{thr}/RecoClustersMatchedSimClustersEn_Mult RecoClustersPt_Mult fake", 
                    f"'Score{thr}/Fake_vs_PtEta' 'Fake Rate vs p_{{T}}-#eta' Score{thr}/RecoClustersMatchedSimClustersPt_Eta RecoClustersPt_Eta fake",       
                    f"'Score{thr}/Fake_vs_PtPhi' 'Fake Rate vs p_{{T}}-#phi' Score{thr}/RecoClustersMatchedSimClustersPt_Phi RecoClustersPt_Phi fake",       
                    f"'Score{thr}/Fake_vs_PtMult' 'Fake Rate vs p_{{T}}-Mult' Score{thr}/RecoClustersMatchedSimClustersPt_Mult RecoClustersPt_Mult fake",    
                    f"'Score{thr}/Fake_vs_MultEta' 'Fake Rate vs Mult-#eta' Score{thr}/RecoClustersMatchedSimClustersMult_Eta RecoClustersMult_Eta fake",    
                    f"'Score{thr}/Fake_vs_MultPhi' 'Fake Rate vs Mult-#phi' Score{thr}/RecoClustersMatchedSimClustersMult_Phi RecoClustersMult_Phi fake",    
            )
            ],
    ),
    efficiencyProfile = cms.untracked.vstring( # for smoother rebinning
        *[ item
            for thr in _thresholds
            for name, suf in zip(('', '_Reconstructable'), ('', 'Reconstructable'))
            for item in (
                    # Efficiency
                    f"'Score{thr}/Eff_vs_En{name}' 'Efficiency vs Energy {suf}' Score{thr}/SimClustersMatchedRecoClustersEn SimClusters{suf}En",
                    f"'Score{thr}/Eff_vs_EnHits{name}' 'Efficiency vs Hits Energy {suf}' Score{thr}/SimClustersMatchedRecoClustersEnHits SimClusters{suf}EnHits",
                    f"'Score{thr}/Eff_vs_EnFrac{name}' 'Efficiency vs Energy fraction {suf}' Score{thr}/SimClustersMatchedRecoClustersEnFrac SimClusters{suf}EnFrac",
                    f"'Score{thr}/Eff_vs_Pt{name}' 'Efficiency vs p_{{T}} {suf}' Score{thr}/SimClustersMatchedRecoClustersPt SimClusters{suf}Pt",
                    f"'Score{thr}/Eff_vs_Eta{name}' 'Efficiency vs #eta {suf}' Score{thr}/SimClustersMatchedRecoClustersEta SimClusters{suf}Eta",
                    f"'Score{thr}/Eff_vs_Phi{name}' 'Efficiency vs #phi {suf}' Score{thr}/SimClustersMatchedRecoClustersPhi SimClusters{suf}Phi",
                    f"'Score{thr}/Eff_vs_Mult{name}' 'Efficiency vs Multiplicity {suf}' Score{thr}/SimClustersMatchedRecoClustersMult SimClusters{suf}Mult",
                    # Split rate
                    f"'Score{thr}/Split_vs_En{name}' 'Split Rate vs Energy {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEn SimClusters{suf}En",             
                    f"'Score{thr}/Split_vs_EnHits{name}' 'Split Rate vs Hits Energy {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEnHits SimClusters{suf}EnHits",             
                    f"'Score{thr}/Split_vs_EnFrac{name}' 'Split Rate vs Energy fraction {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEnFrac SimClusters{suf}EnFrac",             
                    f"'Score{thr}/Split_vs_Pt{name}' 'Split Rate vs p_{{T}} {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersPt SimClusters{suf}Pt",            
                    f"'Score{thr}/Split_vs_Eta{name}' 'Split Rate vs #eta {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEta SimClusters{suf}Eta",            
                    f"'Score{thr}/Split_vs_Phi{name}' 'Split Rate vs #phi {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersPhi SimClusters{suf}Phi",            
                    f"'Score{thr}/Split_vs_Mult{name}' 'Split Rate vs Multiplicity {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersMult SimClusters{suf}Mult", 
            )
          ],
        *[ item
            for thr in _thresholds
            for item in (
                    # Fake rate
                    f"'Score{thr}/Fake_vs_En' 'Fake Rate vs Energy' Score{thr}/RecoClustersMatchedSimClustersEn RecoClustersEn fake",
                    f"'Score{thr}/Fake_vs_Pt' 'Fake Rate vs p_{{T}}' Score{thr}/RecoClustersMatchedSimClustersPt RecoClustersPt fake",
                    f"'Score{thr}/Fake_vs_Eta' 'Fake Rate vs #eta' Score{thr}/RecoClustersMatchedSimClustersEta RecoClustersEta fake",
                    f"'Score{thr}/Fake_vs_Phi' 'Fake Rate vs #phi' Score{thr}/RecoClustersMatchedSimClustersPhi RecoClustersPhi fake",
                    f"'Score{thr}/Fake_vs_Mult' 'Fake Rate vs Multiplicity' Score{thr}/RecoClustersMatchedSimClustersMult RecoClustersMult fake", 
                    # Merge rate
                    f"'Score{thr}/Merge_vs_En' 'Merge Rate vs Energy' Score{thr}/RecoClustersMultiMatchedSimClustersEn RecoClustersEn",
                    f"'Score{thr}/Merge_vs_Pt' 'Merge Rate vs p_{{T}}' Score{thr}/RecoClustersMultiMatchedSimClustersPt RecoClustersPt",
                    f"'Score{thr}/Merge_vs_Eta' 'Merge Rate vs #eta' Score{thr}/RecoClustersMultiMatchedSimClustersEta RecoClustersEta",
                    f"'Score{thr}/Merge_vs_Phi' 'Merge Rate vs #phi' Score{thr}/RecoClustersMultiMatchedSimClustersPhi RecoClustersPhi",
                    f"'Score{thr}/Merge_vs_Mult' 'Merge Rate vs Mult' Score{thr}/RecoClustersMultiMatchedSimClustersMult RecoClustersMult",
            )
          ],
    ),
    resolution = cms.vstring(),
    resolutionProfile = cms.untracked.vstring(
        *[ item
            for thr in _thresholds
            for item in (
                    f"'Score{thr}/ResponseE_En' 'Response vs Energy' Score{thr}/ResponseE_En rms",
                    f"'Score{thr}/ResponseE_EnHits' 'Response vs Hits Energy' Score{thr}/ResponseE_EnHits rms",
                    f"'Score{thr}/ResponseE_EnFrac' 'Response vs Energy fraction' Score{thr}/ResponseE_EnFrac rms",
                    f"'Score{thr}/ResponseE_Pt' 'Response vs  p_{{T}}' Score{thr}/ResponseE_Pt rms",
                    f"'Score{thr}/ResponseE_Eta' 'Response vs #eta' Score{thr}/ResponseE_Eta rms",
                    f"'Score{thr}/ResponseE_Phi' 'Response vs #phi' Score{thr}/ResponseE_Phi rms",
                    f"'Score{thr}/ResponseE_Mult' 'Response vs Multiplicity' Score{thr}/ResponseE_Mult rms"
            )
            ],
        ),
    verbose = cms.untracked.uint32(2), 
    outputFileName = cms.untracked.string("")
)
