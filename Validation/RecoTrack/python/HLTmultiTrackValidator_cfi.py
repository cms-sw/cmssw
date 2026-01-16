import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.associators_cff import *

from Validation.RecoTrack.MultiTrackValidator_cfi import *
hltMultiTrackValidator = multiTrackValidator.clone()
hltMultiTrackValidator.ignoremissingtkcollection = cms.bool(True)
hltMultiTrackValidator.dirName = 'HLT/Tracking/ValidationWRTtp/'
hltMultiTrackValidator.label   = ['hltPixelTracks']
hltMultiTrackValidator.beamSpot = 'hltOnlineBeamSpot'
hltMultiTrackValidator.ptMinTP  =  0.4
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.ptMin = 0.4
hltMultiTrackValidator.parametersDefiner = 'hltLhcParametersDefinerForTP'
### for fake rate vs dR ###
hltMultiTrackValidator.calculateDrSingleCollection = False
hltMultiTrackValidator.ignoremissingtrackcollection = cms.untracked.bool(True)

hltMultiTrackValidator.UseAssociators = True
hltMultiTrackValidator.associators = ['hltTrackAssociatorByHits']
