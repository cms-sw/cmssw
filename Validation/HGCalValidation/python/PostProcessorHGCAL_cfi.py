import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge

maxlayerzm =  50# last layer of BH -z
maxlayerzp =  100# last layer of BH +z

#hgcalLayerClusters
eff_layers = ["effic_eta_layer{:02d} 'LayerCluster Efficiency vs #eta Layer{:02d} in z-' Num_CaloParticle_Eta_perlayer{:02d} Denom_CaloParticle_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "effic_eta_layer{:02d} 'LayerCluster Efficiency vs #eta Layer{:02d} in z+' Num_CaloParticle_Eta_perlayer{:02d} Denom_CaloParticle_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ]
eff_layers.extend(["effic_phi_layer{:02d} 'LayerCluster Efficiency vs #phi Layer{:02d} in z-' Num_CaloParticle_Phi_perlayer{:02d} Denom_CaloParticle_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "effic_phi_layer{:02d} 'LayerCluster Efficiency vs #phi Layer{:02d} in z+' Num_CaloParticle_Phi_perlayer{:02d} Denom_CaloParticle_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["duplicate_eta_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #eta Layer{:02d} in z-' NumDup_CaloParticle_Eta_perlayer{:02d} Denom_CaloParticle_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "duplicate_eta_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #eta Layer{:02d} in z+' NumDup_CaloParticle_Eta_perlayer{:02d} Denom_CaloParticle_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["duplicate_phi_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #phi Layer{:02d} in z-' NumDup_CaloParticle_Phi_perlayer{:02d} Denom_CaloParticle_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "duplicate_phi_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #phi Layer{:02d} in z+' NumDup_CaloParticle_Phi_perlayer{:02d} Denom_CaloParticle_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["fake_eta_layer{:02d} 'LayerCluster Fake Rate vs #eta Layer{:02d} in z-' Num_LayerCluster_Eta_perlayer{:02d} Denom_LayerCluster_Eta_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "fake_eta_layer{:02d} 'LayerCluster Fake Rate vs #eta Layer{:02d} in z+' Num_LayerCluster_Eta_perlayer{:02d} Denom_LayerCluster_Eta_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["fake_phi_layer{:02d} 'LayerCluster Fake Rate vs #phi Layer{:02d} in z-' Num_LayerCluster_Phi_perlayer{:02d} Denom_LayerCluster_Phi_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "fake_phi_layer{:02d} 'LayerCluster Fake Rate vs #phi Layer{:02d} in z+' Num_LayerCluster_Phi_perlayer{:02d} Denom_LayerCluster_Phi_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["merge_eta_layer{:02d} 'LayerCluster Merge Rate vs #eta Layer{:02d} in z-' NumMerge_LayerCluster_Eta_perlayer{:02d} Denom_LayerCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "merge_eta_layer{:02d} 'LayerCluster Merge Rate vs #eta Layer{:02d} in z+' NumMerge_LayerCluster_Eta_perlayer{:02d} Denom_LayerCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_layers.extend(["merge_phi_layer{:02d} 'LayerCluster Merge Rate vs #phi Layer{:02d} in z-' NumMerge_LayerCluster_Phi_perlayer{:02d} Denom_LayerCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "merge_phi_layer{:02d} 'LayerCluster Merge Rate vs #phi Layer{:02d} in z+' NumMerge_LayerCluster_Phi_perlayer{:02d} Denom_LayerCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])

postProcessorHGCALlayerclusters= DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring('HGCAL/HGCalValidator/hgcalLayerClusters/LCtoCP_association'),
    efficiency = cms.vstring(eff_layers),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))

#SimClusters
eff_simclusters = ["effic_eta_layer{:02d} 'LayerCluster Efficiency vs #eta Layer{:02d} in z-' Num_SimCluster_Eta_perlayer{:02d} Denom_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "effic_eta_layer{:02d} 'LayerCluster Efficiency vs #eta Layer{:02d} in z+' Num_SimCluster_Eta_perlayer{:02d} Denom_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ]
eff_simclusters.extend(["effic_phi_layer{:02d} 'LayerCluster Efficiency vs #phi Layer{:02d} in z-' Num_SimCluster_Phi_perlayer{:02d} Denom_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "effic_phi_layer{:02d} 'LayerCluster Efficiency vs #phi Layer{:02d} in z+' Num_SimCluster_Phi_perlayer{:02d} Denom_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["duplicate_eta_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #eta Layer{:02d} in z-' NumDup_SimCluster_Eta_perlayer{:02d} Denom_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "duplicate_eta_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #eta Layer{:02d} in z+' NumDup_SimCluster_Eta_perlayer{:02d} Denom_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["duplicate_phi_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #phi Layer{:02d} in z-' NumDup_SimCluster_Phi_perlayer{:02d} Denom_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "duplicate_phi_layer{:02d} 'LayerCluster Duplicate(Split) Rate vs #phi Layer{:02d} in z+' NumDup_SimCluster_Phi_perlayer{:02d} Denom_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["fake_eta_layer{:02d} 'LayerCluster Fake Rate vs #eta Layer{:02d} in z-' Num_LayerCluster_in_SimCluster_Eta_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Eta_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "fake_eta_layer{:02d} 'LayerCluster Fake Rate vs #eta Layer{:02d} in z+' Num_LayerCluster_in_SimCluster_Eta_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Eta_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["fake_phi_layer{:02d} 'LayerCluster Fake Rate vs #phi Layer{:02d} in z-' Num_LayerCluster_in_SimCluster_Phi_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Phi_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "fake_phi_layer{:02d} 'LayerCluster Fake Rate vs #phi Layer{:02d} in z+' Num_LayerCluster_in_SimCluster_Phi_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Phi_perlayer{:02d} fake".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["merge_eta_layer{:02d} 'LayerCluster Merge Rate vs #eta Layer{:02d} in z-' NumMerge_LayerCluster_in_SimCluster_Eta_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "merge_eta_layer{:02d} 'LayerCluster Merge Rate vs #eta Layer{:02d} in z+' NumMerge_LayerCluster_in_SimCluster_Eta_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Eta_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])
eff_simclusters.extend(["merge_phi_layer{:02d} 'LayerCluster Merge Rate vs #phi Layer{:02d} in z-' NumMerge_LayerCluster_in_SimCluster_Phi_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) if (i<maxlayerzm) else "merge_phi_layer{:02d} 'LayerCluster Merge Rate vs #phi Layer{:02d} in z+' NumMerge_LayerCluster_in_SimCluster_Phi_perlayer{:02d} Denom_LayerCluster_in_SimCluster_Phi_perlayer{:02d}".format(i, i%maxlayerzm+1, i, i) for i in range(maxlayerzp) ])

subdirsSim = ['HGCAL/HGCalValidator/simClusters/ticlTracksters'+iteration+'/' for iteration in ticlIterLabelsMerge]
subdirsSim.extend(['HGCAL/HGCalValidator/simClusters/ticlSimTracksters'])
postProcessorHGCALsimclusters= DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(subdirsSim),
    efficiency = cms.vstring(eff_simclusters),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))

eff_multiclusters = ["effic_eta 'MultiCluster Efficiency vs #eta' Num_CaloParticle_Eta Denom_CaloParticle_Eta"]
eff_multiclusters.extend(["effic_phi 'MultiCluster Efficiency vs #phi' Num_CaloParticle_Phi Denom_CaloParticle_Phi"])
eff_multiclusters.extend(["duplicate_eta 'MultiCluster Duplicate(Split) Rate vs #eta' NumDup_MultiCluster_Eta Denom_MultiCluster_Eta"])
eff_multiclusters.extend(["duplicate_phi 'MultiCluster Duplicate(Split) Rate vs #phi' NumDup_MultiCluster_Phi Denom_MultiCluster_Phi"])
eff_multiclusters.extend(["fake_eta 'MultiCluster Fake Rate vs #eta' Num_MultiCluster_Eta Denom_MultiCluster_Eta fake"])
eff_multiclusters.extend(["fake_phi 'MultiCluster Fake Rate vs #phi'  Num_MultiCluster_Phi Denom_MultiCluster_Phi fake"])
eff_multiclusters.extend(["merge_eta 'MultiCluster Merge Rate vs #eta' NumMerge_MultiCluster_Eta Denom_MultiCluster_Eta"])
eff_multiclusters.extend(["merge_phi 'MultiCluster Merge Rate vs #phi' NumMerge_MultiCluster_Phi Denom_MultiCluster_Phi"])

subdirsMult = ['HGCAL/HGCalValidator/hgcalMultiClusters/','HGCAL/HGCalValidator/ticlMultiClustersFromSimTracksters/']
subdirsMult.extend('HGCAL/HGCalValidator/ticlMultiClustersFromTracksters'+iteration+'/' for iteration in ticlIterLabelsMerge)

postProcessorHGCALmulticlusters = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subdirsMult),
  efficiency = cms.vstring(eff_multiclusters),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)
