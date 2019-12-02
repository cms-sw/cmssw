autoValidation = { 'liteTracking' : ['prevalidationLiteTracking','validationLiteTracking','validationHarvesting'],
                   'trackingOnlyValidation' : ['globalPrevalidationTrackingOnly','globalValidationTrackingOnly','postValidation_trackingOnly'],
                   'pixelTrackingOnlyValidation' : ['globalPrevalidationPixelTrackingOnly','globalValidationPixelTrackingOnly','postValidation_trackingOnly'],
                   'trackingValidation': ['globalPrevalidationTracking','globalValidationTrackingOnly','postValidationTracking'],
                   'muonOnlyValidation' : ['globalPrevalidationMuons','globalValidationMuons','postValidation_muons'],
                   'bTagOnlyValidation' : ['prebTagSequenceMC','bTagPlotsMCbcl','bTagCollectorSequenceMCbcl'],
                   'JetMETOnlyValidation' : ['globalPrevalidationJetMETOnly','globalValidationJetMETonly','postValidation_JetMET'],
                   'electronOnlyValidation' : ['', 'electronValidationSequence', 'electronPostValidationSequence'],
                   'photonOnlyValidation' : ['', 'photonValidationSequence', 'photonPostProcessor'],
                   'tauOnlyValidation' : ['produceDenoms', 'pfTauRunDQMValidation', 'runTauEff'],
                   'hcalOnlyValidation' : ['globalPrevalidationHCAL','globalValidationHCAL','postValidation_HCAL'],
                   'baseValidation' : ['baseCommonPreValidation','baseCommonValidation','postValidation_common'],
                   'miniAODValidation' : ['prevalidationMiniAOD','validationMiniAOD','validationHarvestingMiniAOD'],
                   'standardValidation' : ['prevalidation','validation','validationHarvesting'],
                   'standardValidationNoHLT' : ['prevalidationNoHLT','validationNoHLT','validationHarvestingNoHLT'],
                   'HGCalValidation' : ['', 'globalValidationHGCal', 'hgcalValidatorPostProcessor'],
                   'MTDValidation' : ['', 'globalValidationMTD', 'mtdValidationPostProcessor'],
                   'OuterTrackerValidation' : ['', 'globalValidationOuterTracker', 'postValidationOuterTracker'],
                   'ecalValidation_phase2' : ['', 'validationECALPhase2', ''],
                   'TrackerPhase2Validation' : ['', 'trackerphase2ValidationSource', 'trackerphase2ValidationHarvesting'],
                 }

_phase2_allowed = ['baseValidation','trackingValidation','muonOnlyValidation','JetMETOnlyValidation', 'electronOnlyValidation', 'photonOnlyValidation','bTagOnlyValidation', 'tauOnlyValidation', 'hcalOnlyValidation', 'HGCalValidation', 'MTDValidation', 'OuterTrackerValidation', 'ecalValidation_phase2', 'TrackerPhase2Validation']
autoValidation['phase2Validation'] = ['','','']
for i in range(0,3):
    autoValidation['phase2Validation'][i] = '+'.join([_f for _f in [autoValidation[m][i] for m in _phase2_allowed] if _f])
