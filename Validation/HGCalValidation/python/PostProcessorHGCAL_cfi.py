import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

eff_layers = ["effic_eta_layer%d 'LayerCluster Efficiency vs #eta Layer%d' Num_CaloParticle_Eta_perlayer%d Denom_CaloParticle_Eta_perlayer%d" % (i, i, i, i)  for i in range(1,53) ]
eff_layers.extend(["effic_phi_layer%d 'LayerCluster Efficiency vs #phi Layer%d' Num_CaloParticle_Phi_perlayer%d Denom_CaloParticle_Phi_perlayer%d" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["duplicate_eta_layer%d 'LayerCluster Duplicate(Split) Rate vs #eta Layer%d' NumDup_CaloParticle_Eta_perlayer%d Denom_CaloParticle_Eta_perlayer%d" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["duplicate_phi_layer%d 'LayerCluster Duplicate(Split) Rate vs #phi Layer%d' NumDup_CaloParticle_Phi_perlayer%d Denom_CaloParticle_Phi_perlayer%d" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["fake_eta_layer%d 'LayerCluster Fake Rate vs #eta Layer%d' Num_LayerCluster_Eta_perlayer%d Denom_LayerCluster_Eta_perlayer%d fake" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["fake_phi_layer%d 'LayerCluster Fake Rate vs #phi Layer%d' Num_LayerCluster_Phi_perlayer%d Denom_LayerCluster_Phi_perlayer%d fake" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["merge_eta_layer%d 'LayerCluster Merge Rate vs #eta Layer%d' NumMerge_LayerCluster_Eta_perlayer%d Denom_LayerCluster_Eta_perlayer%d" % (i, i, i, i)  for i in range(1,53) ])
eff_layers.extend(["merge_phi_layer%d 'LayerCluster Merge Rate vs #phi Layer%d' NumMerge_LayerCluster_Phi_perlayer%d Denom_LayerCluster_Phi_perlayer%d" % (i, i, i, i)  for i in range(1,53) ])

postProcessorHGCAL = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring('HGCAL/HGCalValidator/hgcalLayerClusters/'),
    efficiency = cms.vstring(eff_layers),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))

