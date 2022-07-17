import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.associators_cff import *

from Validation.RecoTrack.MultiTrackValidator_cfi import *
hltMultiTrackValidator = multiTrackValidator.clone()
hltMultiTrackValidator.ignoremissingtkcollection = cms.bool(True)
hltMultiTrackValidator.dirName = 'HLT/Tracking/ValidationWRTtp/'
hltMultiTrackValidator.label   = ['hltPixelTracks']
hltMultiTrackValidator.beamSpot = 'hltOnlineBeamSpot'
hltMultiTrackValidator.ptMinTP  =  0.4
hltMultiTrackValidator.lipTP    = 30.0
hltMultiTrackValidator.tipTP    = 60.0
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.ptMin =  0.9
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.lip   = 30.0
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.tip   = 60.0
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt   = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone(ptMin=0.4)
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.parametersDefiner = 'hltLhcParametersDefinerForTP'
### for fake rate vs dR ###
hltMultiTrackValidator.calculateDrSingleCollection = False
hltMultiTrackValidator.ignoremissingtrackcollection = cms.untracked.bool(True)

hltMultiTrackValidator.UseAssociators = True
hltMultiTrackValidator.associators = ['hltTrackAssociatorByHits']
