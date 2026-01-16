import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from Validation.HGCalValidation.BarrelValidator_cff import barrelValidator

prefix = 'BarrelCalorimeters/BarrelValidator/'
maxlayer = barrelValidator.totallayers_to_monitor.value()

#barrelLayerClusters
eff_layers = ["effic_eta_layer_barrel{} 'LayerCluster Efficiency vs #eta Layer{}' Num_CaloParticle_Eta_perlayer_barrel{} Denom_CaloParticle_Eta_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)]
eff_layers.extend(["effic_phi_layer_barrel{} 'LayerCluster Efficiency vs #phi Layer{}' Num_CaloParticle_Phi_perlayer_barrel{} Denom_CaloParticle_Phi_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["duplicate_eta_layer_barrel{} 'LayerCluster Duplicate(Split) Rate vs #eta Layer{}' NumDup_CaloParticle_Eta_perlayer_barrel{} Denom_CaloParticle_Eta_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["duplicate_phi_layer_barrel{} 'LayerCluster Duplicate(Split) Rate vs #phi Layer{}' NumDup_CaloParticle_Phi_perlayer_barrel{} Denom_CaloParticle_Phi_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["fake_eta_layer_barrel{} 'LayerCluster Fake Rate vs #eta Layer{}' Num_LayerCluster_Eta_perlayer_barrel{} Denom_LayerCluster_Eta_perlayer_barrel{} fake".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["fake_phi_layer_barrel{} 'LayerCluster Fake Rate vs #phi Layer{}' Num_LayerCluster_Phi_perlayer_barrel{} Denom_LayerCluster_Phi_perlayer_barrel{} fake".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["merge_eta_layer_barrel{} 'LayerCluster Merge Rate vs #eta Layer{}' NumMerge_LayerCluster_Eta_perlayer_barrel{} Denom_LayerCluster_Eta_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)])
eff_layers.extend(["merge_phi_layer_barrel{} 'LayerCluster Merge Rate vs #phi Layer{}' NumMerge_LayerCluster_Phi_perlayer_barrel{} Denom_LayerCluster_Phi_perlayer_barrel{}".format(i, i, i, i) for i in range(maxlayer)])

lcToCP_linking = barrelValidator.label_LCToCPLinking.value()
postProcessorBarrellayerclusters = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(prefix + barrelValidator.label_layerClustersPlots.value() + '/' + lcToCP_linking),
    efficiency = cms.vstring(eff_layers),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))

