import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsMerge
from Validation.HGCalValidation.HGCalValidator_cfi import hgcalValidator

prefix = 'HGCAL/HGCalValidator/'
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

lcToCP_linking = hgcalValidator.label_LCToCPLinking._InputTag__moduleLabel
postProcessorHGCALlayerclusters= DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(prefix + hgcalValidator.label_layerClusterPlots._InputTag__moduleLabel + '/' + lcToCP_linking),
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

subdirsSim = [prefix + hgcalValidator.label_SimClusters._InputTag__moduleLabel + '/ticlTracksters'+iteration+'/' for iteration in ticlIterLabelsMerge]
subdirsSim.extend([prefix + hgcalValidator.label_SimClusters._InputTag__moduleLabel + '/ticlSimTracksters'])
postProcessorHGCALsimclusters= DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(subdirsSim),
    efficiency = cms.vstring(eff_simclusters),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))

eff_tracksters = ["purity_eta 'Trackster Purity vs #eta' Num_CaloParticle_Eta Denom_CaloParticle_Eta"]
eff_tracksters.extend(["purity_phi 'Trackster Purity vs #phi' Num_CaloParticle_Phi Denom_CaloParticle_Phi"])
eff_tracksters.extend(["effic_eta 'Trackster Efficiency vs #eta' NumEff_CaloParticle_Eta Denom_CaloParticle_Eta"])
eff_tracksters.extend(["effic_phi 'Trackster Efficiency vs #phi' NumEff_CaloParticle_Phi Denom_CaloParticle_Phi"])
eff_tracksters.extend(["duplicate_eta 'Trackster Duplicate(Split) Rate vs #eta' NumDup_Trackster_Eta Denom_Trackster_Eta"])
eff_tracksters.extend(["duplicate_phi 'Trackster Duplicate(Split) Rate vs #phi' NumDup_Trackster_Phi Denom_Trackster_Phi"])
eff_tracksters.extend(["fake_eta 'Trackster Fake Rate vs #eta' Num_Trackster_Eta Denom_Trackster_Eta fake"])
eff_tracksters.extend(["fake_phi 'Trackster Fake Rate vs #phi'  Num_Trackster_Phi Denom_Trackster_Phi fake"])
eff_tracksters.extend(["merge_eta 'Trackster Merge Rate vs #eta' NumMerge_Trackster_Eta Denom_Trackster_Eta"])
eff_tracksters.extend(["merge_phi 'Trackster Merge Rate vs #phi' NumMerge_Trackster_Phi Denom_Trackster_Phi"])

tsToCP_linking = hgcalValidator.label_TSToCPLinking._InputTag__moduleLabel
subdirsTracksters = [prefix+'hgcalTracksters/'+tsToCP_linking, prefix+'ticlSimTracksters/'+tsToCP_linking]
subdirsTracksters.extend(prefix+'ticlTracksters'+iteration+'/'+tsToCP_linking for iteration in ticlIterLabelsMerge)

postProcessorHGCALTracksters = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subdirsTracksters),
  efficiency = cms.vstring(eff_tracksters),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)
