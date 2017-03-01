import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
hltLhcParametersDefinerForTP = LhcParametersDefinerForTP.clone()
hltLhcParametersDefinerForTP.ComponentName = cms.string('hltLhcParametersDefinerForTP')
hltLhcParametersDefinerForTP.beamSpot      = cms.untracked.InputTag('hltOnlineBeamSpot')

from Validation.RecoTrack.associators_cff import *

from Validation.RecoTrack.MultiTrackValidator_cfi import *
hltMultiTrackValidator = multiTrackValidator.clone()
hltMultiTrackValidator.ignoremissingtkcollection = cms.bool(True)
hltMultiTrackValidator.dirName = cms.string('HLT/Tracking/ValidationWRTtp/')
hltMultiTrackValidator.label   = cms.VInputTag(cms.InputTag("hltPixelTracks"))
hltMultiTrackValidator.beamSpot = cms.InputTag("hltOnlineBeamSpot")
hltMultiTrackValidator.ptMinTP  = cms.double( 0.4)
hltMultiTrackValidator.lipTP    = cms.double(35.0)
hltMultiTrackValidator.tipTP    = cms.double(70.0)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.ptMin = cms.double( 0.4)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.lip   = cms.double(35.0)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.tip   = cms.double(70.0)
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt   = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.parametersDefiner = cms.string('hltLhcParametersDefinerForTP')
#hltMultiTrackValidator.parametersDefiner = cms.string('LhcParametersDefinerForTP')
### for fake rate vs dR ###
hltMultiTrackValidator.calculateDrSingleCollection = False
hltMultiTrackValidator.ignoremissingtrackcollection = cms.untracked.bool(True)

hltMultiTrackValidator.UseAssociators = True
hltMultiTrackValidator.associators = ['hltTrackAssociatorByHits']
