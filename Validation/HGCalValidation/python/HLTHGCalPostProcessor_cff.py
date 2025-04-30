import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALlayerclusters as _postProcessorHGCALlayerclusters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALsimclusters as _postProcessorHGCALsimclusters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALTracksters as _postProcessorHGCALTracksters
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCALCandidates as _postProcessorHGCALCandidates 

from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels as _hltTiclIterLabels
from Validation.HGCalValidation.HLTHGCalValidator_cff import hltHgcalValidator as _hltHgcalValidator

hltPrefix = 'HLT/HGCAL/HGCalValidator/'
hltTracksterLabels = _hltTiclIterLabels.copy()
hltTracksterLabels.extend(['hltTiclSimTracksters', 'hltTiclSimTracksters_fromCPs'])

hltLcToCP_linking = _hltHgcalValidator.label_LCToCPLinking._InputTag__moduleLabel
hltPostProcessorHGCALlayerclusters = _postProcessorHGCALlayerclusters.clone(
    subDirs = cms.untracked.vstring(hltPrefix + _hltHgcalValidator.label_layerClusterPlots._InputTag__moduleLabel + '/' + hltLcToCP_linking),
)

hltSubdirsSim = [hltPrefix + _hltHgcalValidator.label_SimClusters._InputTag__moduleLabel + '/'+iteration+'/' for iteration in hltTracksterLabels]
hltPostProcessorHGCALsimclusters = _postProcessorHGCALsimclusters.clone(
    subDirs = cms.untracked.vstring(hltSubdirsSim)
)

hltTSbyHits_CP = _hltHgcalValidator.label_TSbyHitsCP.value()
hltSubdirsTracksters = [hltPrefix+iteration+'/'+hltTSbyHits_CP for iteration in hltTracksterLabels]

hltTSbyLCs = _hltHgcalValidator.label_TSbyLCs.value()
hltSubdirsTracksters.extend(hltPrefix+iteration+'/'+hltTSbyLCs for iteration in hltTracksterLabels)

hltTSbyLCs_CP = _hltHgcalValidator.label_TSbyLCsCP.value()
hltSubdirsTracksters.extend(hltPrefix+iteration+'/'+hltTSbyLCs_CP for iteration in hltTracksterLabels)

hltTSbyHits = _hltHgcalValidator.label_TSbyHits.value()
hltSubdirsTracksters.extend(hltPrefix+iteration+'/'+hltTSbyHits for iteration in hltTracksterLabels)

hltPostProcessorHGCALTracksters = _postProcessorHGCALTracksters.clone(
    subDirs = cms.untracked.vstring(hltSubdirsTracksters)
)

hltNeutrals = ["photons", "neutral_pions", "neutral_hadrons"]
hltCharged = ["electrons", "muons", "charged_hadrons"]
hltSubDirsCandidates = [hltPrefix + _hltHgcalValidator.ticlCandidates.value() + "/" + c for cands in (hltNeutrals, hltCharged) for c in cands]

hltPostProcessorHGCALCandidates = _postProcessorHGCALCandidates.clone(
    subDirs = cms.untracked.vstring(hltSubDirsCandidates)
)

hltHcalValidatorPostProcessor = cms.Sequence(
    hltPostProcessorHGCALlayerclusters+
    hltPostProcessorHGCALsimclusters+
    hltPostProcessorHGCALTracksters+
    hltPostProcessorHGCALCandidates        
)
