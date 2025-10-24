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
                f"'Score{thr}/Fake_vs_EnEta{name}' 'Fake Rate vs Energy-#eta {suf}' Score{thr}/RecoClustersMatchedSimClustersEn_Eta RecoClusters{suf}Pt_Eta fake",    
                f"'Score{thr}/Fake_vs_EnPhi{name}' 'Fake Rate vs Energy-#phi {suf}' Score{thr}/RecoClustersMatchedSimClustersEn_Phi RecoClusters{suf}Pt_Phi fake",    
                f"'Score{thr}/Fake_vs_EnMult{name}' 'Fake Rate vs Energy-Mult {suf}' Score{thr}/RecoClustersMatchedSimClustersEn_Mult RecoClusters{suf}Pt_Mult fake", 
                f"'Score{thr}/Fake_vs_PtEta{name}' 'Fake Rate vs p_{{T}}-#eta {suf}' Score{thr}/RecoClustersMatchedSimClustersPt_Eta RecoClusters{suf}Pt_Eta fake",       
                f"'Score{thr}/Fake_vs_PtPhi{name}' 'Fake Rate vs p_{{T}}-#phi {suf}' Score{thr}/RecoClustersMatchedSimClustersPt_Phi RecoClusters{suf}Pt_Phi fake",       
                f"'Score{thr}/Fake_vs_PtMult{name}' 'Fake Rate vs p_{{T}}-Mult {suf}' Score{thr}/RecoClustersMatchedSimClustersPt_Mult RecoClusters{suf}Pt_Mult fake",    
                f"'Score{thr}/Fake_vs_MultEta{name}' 'Fake Rate vs Mult-#eta {suf}' Score{thr}/RecoClustersMatchedSimClustersMult_Eta RecoClusters{suf}Mult_Eta fake",    
                f"'Score{thr}/Fake_vs_MultPhi{name}' 'Fake Rate vs Mult-#phi {suf}' Score{thr}/RecoClustersMatchedSimClustersMult_Phi RecoClusters{suf}Mult_Phi fake",    
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
                   # Fake rate
                   f"'Score{thr}/Fake_vs_En{name}' 'Fake Rate vs Energy {suf}' Score{thr}/RecoClustersMatchedSimClustersEn RecoClusters{suf}En",             
                   f"'Score{thr}/Fake_vs_Pt{name}' 'Fake Rate vs p_{{T}} {suf}' Score{thr}/RecoClustersMatchedSimClustersPt RecoClusters{suf}Pt",            
                   f"'Score{thr}/Fake_vs_Eta{name}' 'Fake Rate vs #eta {suf}' Score{thr}/RecoClustersMatchedSimClustersEta RecoClusters{suf}Eta",            
                   f"'Score{thr}/Fake_vs_Phi{name}' 'Fake Rate vs #phi {suf}' Score{thr}/RecoClustersMatchedSimClustersPhi RecoClusters{suf}Phi",            
                   f"'Score{thr}/Fake_vs_Mult{name}' 'Fake Rate vs Multiplicity {suf}' Score{thr}/RecoClustersMatchedSimClustersMult RecoClusters{suf}Mult", 
                   # Duplicate rate
                   f"'Score{thr}/Dup_vs_En{name}' 'Dup Rate vs Energy {suf}' Score{thr}/RecoClustersMultiMatchedSimClustersEn RecoClusters{suf}En",      
                   f"'Score{thr}/Dup_vs_Pt{name}' 'Dup Rate vs p_{{T}} {suf}' Score{thr}/RecoClustersMultiMatchedSimClustersPt RecoClusters{suf}Pt",     
                   f"'Score{thr}/Dup_vs_Eta{name}' 'Dup Rate vs #eta {suf}' Score{thr}/RecoClustersMultiMatchedSimClustersEta RecoClusters{suf}Eta",     
                   f"'Score{thr}/Dup_vs_Phi{name}' 'Dup Rate vs #phi {suf}' Score{thr}/RecoClustersMultiMatchedSimClustersPhi RecoClusters{suf}Phi",     
                   f"'Score{thr}/Dup_vs_Mult{name}' 'Dup Rate vs Mult {suf}' Score{thr}/RecoClustersMultiMatchedSimClustersMult RecoClusters{suf}Mult",  
                   # Merge rate
                   f"'Score{thr}/Merge_vs_En{name}' 'Merge Rate vs Energy {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEn SimClusters{suf}En",             
                   f"'Score{thr}/Merge_vs_EnHits{name}' 'Merge Rate vs Hits Energy {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEnHits SimClusters{suf}EnHits",             
                   f"'Score{thr}/Merge_vs_EnFrac{name}' 'Merge Rate vs Energy fraction {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEnFrac SimClusters{suf}EnFrac",             
                   f"'Score{thr}/Merge_vs_Pt{name}' 'Merge Rate vs p_{{T}} {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersPt SimClusters{suf}Pt",            
                   f"'Score{thr}/Merge_vs_Eta{name}' 'Merge Rate vs #eta {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersEta SimClusters{suf}Eta",            
                   f"'Score{thr}/Merge_vs_Phi{name}' 'Merge Rate vs #phi {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersPhi SimClusters{suf}Phi",            
                   f"'Score{thr}/Merge_vs_Mult{name}' 'Merge Rate vs Multiplicity {suf}' Score{thr}/SimClustersMultiMatchedRecoClustersMult SimClusters{suf}Mult", 
           )
          ],
    ),
    resolution = cms.vstring(),
    resolutionProfile = cms.untracked.vstring(
        *[ item
           for thr in _thresholds
           for item in (
                   f"'Score{thr}/ResponseEn' 'Response vs Energy' Score{thr}/ResponseEn rms",
                   f"'Score{thr}/ResponseEnHits' 'Response vs Hits Energy' Score{thr}/ResponseEnHits rms",
                   f"'Score{thr}/ResponseEnFrac' 'Response vs Energy fraction' Score{thr}/ResponseEnFrac rms",
                   f"'Score{thr}/ResponsePt' 'Response vs  p_{{T}}' Score{thr}/ResponsePt rms",
                   f"'Score{thr}/ResponseEta' 'Response vs #eta' Score{thr}/ResponseEta rms",
                   f"'Score{thr}/ResponsePhi' 'Response vs #phi' Score{thr}/ResponsePhi rms",
                   f"'Score{thr}/ResponseMult' 'Response vs Multiplicity' Score{thr}/ResponseMult rms"
           )
          ],
        ),
    verbose = cms.untracked.uint32(2), 
    outputFileName = cms.untracked.string("")
)
