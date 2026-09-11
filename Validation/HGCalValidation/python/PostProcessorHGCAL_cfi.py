import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator

__all__ = [
    "tracksterLabels",
    "lcToCP_linking",
    "simDict",
    "TSbyHits_CP",
    "TSbyLCs",
    "TSbyLCs_CP",
    "TSbyHits",
    "variables",
    "postProcessorHGCALlayerclusters",
    "postProcessorHGCALsimclusters",
    "postProcessorHGCALTracksters",
    "postProcessorHGCALCandidates",
]


tracksterLabels = ticlIterLabelsPSet.labels.copy()
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

lcToCP_linking = hgcalValidator.label_LCToCPLinking.value()
postProcessorHGCALlayerclusters = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(prefix + hgcalValidator.label_layerClustersPlots.value() + '/' + lcToCP_linking),
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

subdirsSim = [prefix + hgcalValidator.label_SimClusters.value() + '/'+iteration+'/' for iteration in tracksterLabels]
postProcessorHGCALsimclusters = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring(subdirsSim),
    efficiency = cms.vstring(eff_simclusters),
    resolution = cms.vstring(),
    cumulativeDists = cms.untracked.vstring(),
    noFlowDists = cms.untracked.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4))


trackster_ratios = []
# Must be in sync with labels in HGVHistoProducerAlgo.cc
simDict = {
    "SimTrackster_fromCP_byHits": "_byHits_CP",
    "SimTrackster_byLCs": "_byLCs",
    "SimTrackster_fromCP_byLCs": "_byLCs_CP",
    "SimTrackster_byHits": "_byHits",
}
associations = {association: suffix.removeprefix("_") for association, suffix in simDict.items()}
variables = {
    "eta": "#eta",
    "phi": "#phi",
    "energy": "energy",
    "pt": "p_{T}",
    "R": "R",
    "alpha": "#alpha",
    "time": "time",
}
for association, association_suffix in associations.items():
    for variable, variable_title in variables.items():
        histogram_variable = variable.capitalize()
        trackster_ratios.extend([
            cms.untracked.PSet(
                name=cms.untracked.string(f"purity_{variable}_{association_suffix}"),
                title=cms.untracked.string(f"Trackster Purity vs {variable_title}"),
                numerator=cms.untracked.string(f"Num_{association}_{histogram_variable}"),
                denominator=cms.untracked.string(f"Denom_{association}_{histogram_variable}"),
            ),
            cms.untracked.PSet(
                name=cms.untracked.string(f"effic_{variable}_{association_suffix}"),
                title=cms.untracked.string(f"Trackster Efficiency vs {variable_title}"),
                numerator=cms.untracked.string(f"NumEff_{association}_{histogram_variable}"),
                denominator=cms.untracked.string(f"Denom_{association}_{histogram_variable}"),
            ),
            cms.untracked.PSet(
                name=cms.untracked.string(f"duplicate_{variable}_{association_suffix}"),
                title=cms.untracked.string(f"Trackster Duplicate(Split) vs {variable_title}"),
                numerator=cms.untracked.string(f"NumDup_Trackster_{histogram_variable}_{association_suffix}"),
                denominator=cms.untracked.string(f"Denom_{association}_{histogram_variable}"),
            ),
            cms.untracked.PSet(
                name=cms.untracked.string(f"fake_{variable}_{association_suffix}"),
                title=cms.untracked.string(f"Trackster Fake Rate vs {variable_title}"),
                numerator=cms.untracked.string(f"Num_Trackster_{histogram_variable}_{association_suffix}"),
                denominator=cms.untracked.string(f"Denom_Trackster_{histogram_variable}_{association_suffix}"),
                typeName=cms.untracked.string("fake"),
            ),
            cms.untracked.PSet(
                name=cms.untracked.string(f"merge_{variable}_{association_suffix}"),
                title=cms.untracked.string(f"Trackster Merge Rate vs {variable_title}"),
                numerator=cms.untracked.string(f"NumMerge_Trackster_{histogram_variable}_{association_suffix}"),
                denominator=cms.untracked.string(f"Denom_Trackster_{histogram_variable}_{association_suffix}"),
            ),
        ])


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
  efficiency = cms.vstring(),
  efficiencySets = cms.untracked.VPSet(*trackster_ratios),
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
    for variableName in variables:
        # efficiency
        eff_candidates.append(f"eff_{c}_track_{variableName} '{c.replace('_', ' ')} candidates track efficiency vs {variableName}' num_track_cand_vs_{variableName}_{c} den_cand_vs_{variableName}_{c}")
        eff_candidates.append(f"eff_{c}_pid_{variableName} '{c.replace('_', ' ')} candidates track + pid efficiency vs {variableName}' num_pid_cand_vs_{variableName}_{c} den_cand_vs_{variableName}_{c}")
        eff_candidates.append(f"eff_{c}_energy_{variableName} '{c.replace('_', ' ')} candidates track + pid + energy efficiency vs {variableName}' num_energy_cand_vs_{variableName}_{c} den_cand_vs_{variableName}_{c}")
        # fake
        eff_candidates.append(f"fake_{c}_track_{variableName} '{c.replace('_', ' ')} candidates track fake vs {variableName}' num_fake_track_cand_vs_{variableName}_{c} den_fake_cand_vs_{variableName}_{c}")
        eff_candidates.append(f"fake_{c}_pid_{variableName} '{c.replace('_', ' ')} candidates pid fake vs {variableName}' num_fake_pid_cand_vs_{variableName}_{c} den_fake_cand_vs_{variableName}_{c}")
        eff_candidates.append(f"fake_{c}_energy_{variableName} '{c.replace('_', ' ')} candidates energy fake vs {variableName}' num_fake_energy_cand_vs_{variableName}_{c} den_fake_cand_vs_{variableName}_{c}")
        eff_candidates.append(f"fake_{c}_total_{variableName} '{c.replace('_', ' ')} candidates track + pid + energy fake vs {variableName}' num_fake_total_cand_vs_{variableName}_{c} den_fake_cand_vs_{variableName}_{c}")

for n in neutrals:
    for variableName in variables:
        # efficiency
        eff_candidates.append(f"eff_{n}_pid_{variableName} '{n.replace('_', ' ')} candidates pid efficiency vs {variableName}' num_pid_cand_vs_{variableName}_{n} den_cand_vs_{variableName}_{n}")
        eff_candidates.append(f"eff_{n}_energy_{variableName} '{n.replace('_', ' ')} candidates pid + energy efficiency vs {variableName}' num_energy_cand_vs_{variableName}_{n} den_cand_vs_{variableName}_{n}")
        # fake
        eff_candidates.append(f"fake_{n}_pid_{variableName} '{n.replace('_', ' ')} candidates pid fake vs {variableName}' num_fake_pid_cand_vs_{variableName}_{n} den_fake_cand_vs_{variableName}_{n}")
        eff_candidates.append(f"fake_{n}_energy_{variableName} '{n.replace('_', ' ')} candidates energy fake vs {variableName}' num_fake_energy_cand_vs_{variableName}_{n} den_fake_cand_vs_{variableName}_{n}")
        eff_candidates.append(f"fake_{n}_total_{variableName} '{n.replace('_', ' ')} candidates pid + energy fake vs {variableName}' num_fake_total_cand_vs_{variableName}_{n} den_fake_cand_vs_{variableName}_{n}")


postProcessorHGCALCandidates = DQMEDHarvester('DQMGenericClient',
  subDirs = cms.untracked.vstring(subDirsCandidates),
  efficiency = cms.vstring(eff_candidates),
  resolution = cms.vstring(),
  cumulativeDists = cms.untracked.vstring(),
  noFlowDists = cms.untracked.vstring(),
  outputFileName = cms.untracked.string(""),
  verbose = cms.untracked.uint32(4)
)
