import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Validation.RecoParticleFlow.hltPFValidation_cfi import hltPFClusterTesterECAL
_thresholds = [str(x).replace('.', 'p') for x in hltPFClusterTesterECAL.assocScoreThresholds]

hltPFClusterPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/*"),
    efficiency = cms.vstring(
        *[ item
            for thr in _thresholds
            for item in (
                    f"'Score{thr}/Eff_vs_EnEta' 'Efficiency vs Energy-#eta' Score{thr}/SimClustersMatchedRecoClustersEn_Eta SimClustersEn_Eta",          
                    f"'Score{thr}/Eff_vs_EnPhi' 'Efficiency vs Energy-#phi' Score{thr}/SimClustersMatchedRecoClustersEn_Phi SimClustersEn_Phi",          
                    f"'Score{thr}/Eff_vs_EnMult' 'Efficiency vs Energy-Mult' Score{thr}/SimClustersMatchedRecoClustersEn_Mult SimClustersEn_Mult",       
                    f"'Score{thr}/Eff_vs_EnFracEta' 'Efficiency vs Energy fraction-#eta' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Eta SimClustersEnFrac_Eta",          
                    f"'Score{thr}/Eff_vs_EnFracPhi' 'Efficiency vs Energy fraction-#phi' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Phi SimClustersEnFrac_Phi",          
                    f"'Score{thr}/Eff_vs_EnFracMult' 'Efficiency vs Energy fraction-Mult' Score{thr}/SimClustersMatchedRecoClustersEnFrac_Mult SimClustersEnFrac_Mult",       
                    f"'Score{thr}/Eff_vs_EnSimTrackEta' 'Efficiency vs SimTrack Energy-#eta' Score{thr}/SimClustersMatchedRecoClustersEnSimTrack_Eta SimClustersEnSimTrack_Eta",          
                    f"'Score{thr}/Eff_vs_EnSimTrackPhi' 'Efficiency vs SimTrack Energy-#phi' Score{thr}/SimClustersMatchedRecoClustersEnSimTrack_Phi SimClustersEnSimTrack_Phi",          
                    f"'Score{thr}/Eff_vs_EnSimTrackMult' 'Efficiency vs SimTrack Energy-Mult' Score{thr}/SimClustersMatchedRecoClustersEnSimTrack_Mult SimClustersEnSimTrack_Mult",       
                    f"'Score{thr}/Eff_vs_PtEta' 'Efficiency vs p_{{T}}-#eta' Score{thr}/SimClustersMatchedRecoClustersPt_Eta SimClustersPt_Eta",             
                    f"'Score{thr}/Eff_vs_PtPhi' 'Efficiency vs p_{{T}}-#phi' Score{thr}/SimClustersMatchedRecoClustersPt_Phi SimClustersPt_Phi",             
                    f"'Score{thr}/Eff_vs_PtMult' 'Efficiency vs p_{{T}}-Mult' Score{thr}/SimClustersMatchedRecoClustersPt_Mult SimClustersPt_Mult",          
                    f"'Score{thr}/Eff_vs_MultEta' 'Efficiency vs Mult-#eta' Score{thr}/SimClustersMatchedRecoClustersMult_Eta SimClustersMult_Eta",          
                    f"'Score{thr}/Eff_vs_MultPhi' 'Efficiency vs Mult-#phi' Score{thr}/SimClustersMatchedRecoClustersMult_Phi SimClustersMult_Phi",          
            )
            ],
        *[ item
            for thr in _thresholds
            for item in (    
                    f"'Score{thr}/Fake_vs_EnEta' 'Fake Rate vs Energy-#eta' Score{thr}/RecoClustersMatchedSimClustersEn_Eta RecoClustersEn_Eta fake",    
                    f"'Score{thr}/Fake_vs_EnPhi' 'Fake Rate vs Energy-#phi' Score{thr}/RecoClustersMatchedSimClustersEn_Phi RecoClustersEn_Phi fake",    
                    f"'Score{thr}/Fake_vs_EnMult' 'Fake Rate vs Energy-Mult' Score{thr}/RecoClustersMatchedSimClustersEn_Mult RecoClustersEn_Mult fake", 
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
            for item in (
                    # Efficiency
                    f"'Score{thr}/Eff_vs_En' 'Efficiency vs Energy' Score{thr}/SimClustersMatchedRecoClustersEn SimClustersEn",
                    f"'Score{thr}/Eff_vs_EnFrac' 'Efficiency vs Energy fraction' Score{thr}/SimClustersMatchedRecoClustersEnFrac SimClustersEnFrac",
                    f"'Score{thr}/Eff_vs_EnSimTrack' 'Efficiency vs SimTrack Energy' Score{thr}/SimClustersMatchedRecoClustersEnSimTrack SimClustersEnSimTrack",
                    f"'Score{thr}/Eff_vs_Pt' 'Efficiency vs p_{{T}}' Score{thr}/SimClustersMatchedRecoClustersPt SimClustersPt",
                    f"'Score{thr}/Eff_vs_Eta' 'Efficiency vs #eta' Score{thr}/SimClustersMatchedRecoClustersEta SimClustersEta",
                    f"'Score{thr}/Eff_vs_Phi' 'Efficiency vs #phi' Score{thr}/SimClustersMatchedRecoClustersPhi SimClustersPhi",
                    f"'Score{thr}/Eff_vs_Mult' 'Efficiency vs Multiplicity' Score{thr}/SimClustersMatchedRecoClustersMult SimClustersMult",
                    # Split rate
                    f"'Score{thr}/Split_vs_En' 'Split Rate vs Energy' Score{thr}/SimClustersMultiMatchedRecoClustersEn SimClustersEn",             
                    f"'Score{thr}/Split_vs_EnFrac' 'Split Rate vs Energy fraction' Score{thr}/SimClustersMultiMatchedRecoClustersEnFrac SimClustersEnFrac",             
                    f"'Score{thr}/Split_vs_EnSimTrack' 'Split Rate vs SimTrack Energy' Score{thr}/SimClustersMultiMatchedRecoClustersEnSimTrack SimClustersEnSimTrack",             
                    f"'Score{thr}/Split_vs_Pt' 'Split Rate vs p_{{T}}' Score{thr}/SimClustersMultiMatchedRecoClustersPt SimClustersPt",            
                    f"'Score{thr}/Split_vs_Eta' 'Split Rate vs #eta' Score{thr}/SimClustersMultiMatchedRecoClustersEta SimClustersEta",            
                    f"'Score{thr}/Split_vs_Phi' 'Split Rate vs #phi' Score{thr}/SimClustersMultiMatchedRecoClustersPhi SimClustersPhi",            
                    f"'Score{thr}/Split_vs_Mult' 'Split Rate vs Multiplicity' Score{thr}/SimClustersMultiMatchedRecoClustersMult SimClustersMult", 
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
                    f"'Score{thr}/ResponseE_EnFrac' 'Response vs Energy fraction' Score{thr}/ResponseE_EnFrac rms",
                    f"'Score{thr}/ResponseE_EnSimTrack' 'Response vs SimTrack Energy' Score{thr}/ResponseE_EnSimTrack rms",
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
