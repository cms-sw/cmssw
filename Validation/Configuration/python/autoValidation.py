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
                   'ecalOnlyValidation' : ['globalPrevalidationECALOnly','globalValidationECALOnly','postValidation_ECAL'],
                   'hcalValidation' : ['globalPrevalidationHCAL','globalValidationHCAL','postValidation_HCAL'],
                   'hcalOnlyValidation' : ['globalPrevalidationHCALOnly','globalValidationHCALOnly','postValidation_HCAL'],
                   'baseValidation' : ['baseCommonPreValidation','baseCommonValidation','postValidation_common'],
                   'miniAODValidation' : ['prevalidationMiniAOD','validationMiniAOD','validationHarvestingMiniAOD'],
                   'standardValidation' : ['prevalidation','validation','validationHarvesting'],
                   'standardValidationNoHLT' : ['prevalidationNoHLT','validationNoHLT','validationHarvestingNoHLT'],
                   'hltValidation' : ['prevalidation','hltvalidation','validationHarvesting'],
                   'standardValidationHiMix' : ['prevalidation','validationHiMix','validationHarvesting'],
                   'standardValidationNoHLTHiMix' : ['prevalidationNoHLT','validationNoHLTHiMix','validationHarvestingNoHLT'],
                   'HGCalValidation' : ['globalPrevalidationHGCal', 'globalValidationHGCal', 'hgcalValidatorPostProcessor'],
                   'MTDValidation' : ['', 'globalValidationMTD', 'mtdValidationPostProcessor'],
                   'ecalValidation_phase2' : ['', 'validationECALPhase2', ''],
                   'TrackerPhase2Validation' : ['', 'trackerphase2ValidationSource', 'trackerphase2ValidationHarvesting'],
                 }

_phase2_allowed = ['baseValidation','trackingValidation','muonOnlyValidation','JetMETOnlyValidation', 'electronOnlyValidation', 'photonOnlyValidation','bTagOnlyValidation', 'tauOnlyValidation', 'hcalValidation', 'HGCalValidation', 'MTDValidation', 'ecalValidation_phase2', 'TrackerPhase2Validation', 'hltValidation']
autoValidation['phase2Validation'] = ['','','']
for i in range(0,3):
    autoValidation['phase2Validation'][i] = '+'.join([_f for _f in [autoValidation[m][i] for m in _phase2_allowed] if _f])

# Create autoValidation['phase2ValidationFakeHLT']
_phase2_allowed_fakeHLT = [val for val in _phase2_allowed if val != 'hltValidation']
autoValidation['phase2ValidationFakeHLT'] = ['+'.join([_f for _f in [autoValidation[m][i] for m in _phase2_allowed_fakeHLT] if _f]) for i in range(3)]
