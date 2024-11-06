import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator

tracksterLabels = ticlIterLabels.copy()
tracksterLabels.extend(['ticlSimTracksters', 'ticlSimTracksters_fromCPs'])

prefix = 'HGCAL/HGCalValidator/'
maxlayerzm = hgcalValidator.totallayers_to_monitor.value()# last layer of BH -z
maxlayerzp = 2 * hgcalValidator.totallayers_to_monitor.value()# last layer of BH +z

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
postProcessorHGCALlayerclusters = DQMEDHarvester('DQMGenericClient',
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

subdirsSim = [prefix + hgcalValidator.label_SimClusters._InputTag__moduleLabel + '/'+iteration+'/' for iteration in tracksterLabels]
postProcessorHGCALsimclusters = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(subdirsSim),
    efficiency = cms.vstring(eff_simclusters),
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


TSbyHits_CP = hgcalValidator.label_TSbyHitsCP.value()
subdirsTracksters = [prefix+iteration+'/'+TSbyHits_CP for iteration in tracksterLabels]

TSbyLCs = hgcalValidator.label_TSbyLCs.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyLCs for iteration in tracksterLabels)

TSbyLCs_CP = hgcalValidator.label_TSbyLCsCP.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyLCs_CP for iteration in tracksterLabels)

TSbyHits = hgcalValidator.label_TSbyHits.value()
subdirsTracksters.extend(prefix+iteration+'/'+TSbyHits for iteration in tracksterLabels)



postProcessorHGCALTracksters = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subdirsTracksters),
  efficiency = cms.vstring(eff_tracksters),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)

neutrals = ["photons", "neutral_pions", "neutral_hadrons"]
charged = ["electrons", "muons", "charged_hadrons"]
subDirsCandidates = [prefix + hgcalValidator.ticlCandidates.value() + "/" + c for cands in (neutrals, charged) for c in cands]
eff_candidates = []

for c in charged:
    for var in variables.keys():
        eff_candidates.append("eff_"+c+"_track_"+var+" '"+c.replace("_", " ")+" candidates track efficiency vs "+var+"' num_track_cand_vs_"+var+"_"+c+" den_cand_vs_"+var+"_"+c)
        eff_candidates.append("eff_"+c+"_pid_"+var+" '"+c.replace("_", " ")+" candidates track + pid efficiency vs "+var+"' num_pid_cand_vs_"+var+"_"+c+" den_cand_vs_"+var+"_"+c)
        eff_candidates.append("eff_"+c+"_energy_"+var+" '"+c.replace("_", " ")+" candidates track + pid + energy efficiency vs "+var+"' num_energy_cand_vs_"+var+"_"+c+" den_cand_vs_"+var+"_"+c)
for n in neutrals:
    for var in variables.keys():
        eff_candidates.append("eff_"+n+"_pid_"+var+" '"+n.replace("_", " ")+" candidates pid efficiency vs "+var+"' num_pid_cand_vs_"+var+"_"+n+" den_cand_vs_"+var+"_"+n)
        eff_candidates.append("eff_"+n+"_energy_"+var+" '"+n.replace("_", " ")+" candidates pid + energy efficiency vs "+var+"' num_energy_cand_vs_"+var+"_"+n+" den_cand_vs_"+var+"_"+n)

for c in charged:
    for var in variables.keys():
        eff_candidates.append("fake_"+c+"_track_"+var+" '"+c.replace("_", " ")+" candidates track fake vs "+var+"' num_fake_track_cand_vs_"+var+"_"+c+" den_fake_cand_vs_"+var+"_"+c)
        eff_candidates.append("fake_"+c+"_pid_"+var+" '"+c.replace("_", " ")+" candidates track + pid fake vs "+var+"' num_fake_pid_cand_vs_"+var+"_"+c+" den_fake_cand_vs_"+var+"_"+c)
        eff_candidates.append("fake_"+c+"_energy_"+var+" '"+c.replace("_", " ")+" candidates track + pid + energy fake vs "+var+"' num_fake_energy_cand_vs_"+var+"_"+c+" den_fake_cand_vs_"+var+"_"+c)
for n in neutrals:
    for var in variables.keys():
        eff_candidates.append("fake_"+n+"_pid_"+var+" '"+n.replace("_", " ")+" candidates pid fake vs "+var+"' num_fake_pid_cand_vs_"+var+"_"+n+" den_fake_cand_vs_"+var+"_"+n)
        eff_candidates.append("fake_"+n+"_energy_"+var+" '"+n.replace("_", " ")+" candidates pid + energy fake vs "+var+"' num_fake_energy_cand_vs_"+var+"_"+n+" den_fake_cand_vs_"+var+"_"+n)

postProcessorHGCALCandidates = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subDirsCandidates),
  efficiency = cms.vstring(eff_candidates),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)
