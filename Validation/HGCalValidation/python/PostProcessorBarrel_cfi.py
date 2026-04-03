import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from Validation.HGCalValidation.BarrelValidator_cff import barrelValidator

tracksterLabels = ['ticlTrackstersCLUE3DBarrel']
tracksterLabels.extend(['ticlSimTrackstersBarrel', 'ticlSimTrackstersBarrel_fromCPs'])

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

eff_tracksters = []
# Must be in sync with labels in HGVHistoProducerAlgo.cc
simDict = {"SimTrackster_fromCP_byHits":"_byHits_CP", "SimTrackster_byLCs":"_byLCs", "SimTrackster_fromCP_byLCs":"_byLCs_CP", "SimTrackster_byHits":"_byHits"}
metrics = {"purity":["Purity","_"], "effic":["Efficiency","Eff_"], "fake":["Fake Rate","_"], "duplicate":["Duplicate(Split)","Dup_"], "merge":["Merge Rate","Merge_"]}
variables = {"eta":["#eta",""], "phi":["#phi",""], "energy":["energy"," [GeV]"], "pt":["p_{T}"," [GeV]"]}
for elem in simDict:
    for m in list(metrics.keys())[:2]:
        for v in variables:
            V = v.capitalize()
            eff_tracksters.extend([m+"_"+v+simDict[elem]+" 'Trackster "+metrics[m][0]+" vs "+variables[v][0]+"' Num"+metrics[m][1]+elem+"_"+V+" Denom_"+elem+"_"+V])
    for m in list(metrics.keys())[2:]:
        fakerate = " fake" if (m == "fake") else ""
        for v in variables:
            V = v.capitalize()
            eff_tracksters.extend([m+"_"+v+simDict[elem]+" 'Trackster "+metrics[m][0]+" vs "+variables[v][0]+"' Num"+metrics[m][1]+"Trackster_"+V+simDict[elem]+" Denom_Trackster_"+V+simDict[elem]+fakerate])


TSbyHits_CP = barrelValidator.label_TSbyHitsCP.value()
subdirsTracksters = [prefix+iteration+'/'+TSbyHits_CP for iteration in tracksterLabels]

TSbyLCs = barrelValidator.label_TSbyLCs.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyLCs for iteration in tracksterLabels)

TSbyLCs_CP = barrelValidator.label_TSbyLCsCP.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyLCs_CP for iteration in tracksterLabels)

TSbyHits = barrelValidator.label_TSbyHits.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyHits for iteration in tracksterLabels)

postProcessorBarrelTracksters = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subdirsTracksters),
  efficiency = cms.vstring(eff_tracksters),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)
