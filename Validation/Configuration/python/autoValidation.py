autoValidation = { 'liteTracking' : ['prevalidationLiteTracking','validationLiteTracking','validationHarvesting'],
                   'trackingOnlyValidation' : ['globalPrevalidationTrackingOnly','globalValidationTrackingOnly','postValidation_trackingOnly'],
                   'trackingValidation': ['globalPrevalidationTracking','globalValidationTrackingOnly','postValidationTracking'],
                   'muonOnlyValidation' : ['globalPrevalidationMuons','globalValidationMuons','postValidation_muons'],
                   'bTagOnlyValidation' : ['prebTagSequenceMC','bTagPlotsMCbcl','bTagCollectorSequenceMCbcl'],
                   'JetMETOnlyValidation' : ['globalPrevalidationJetMETOnly','globalValidationJetMETonly','postValidation_JetMET'],
                   'hcalOnlyValidation' : ['globalPrevalidationHCAL','globalValidationHCAL','postValidation_HCAL'],
                   'baseValidation' : ['baseCommonPreValidation','baseCommonValidation','postValidation_common'],
                   'miniAODValidation' : ['prevalidationMiniAOD','validationMiniAOD','validationHarvestingMiniAOD'],
                   'standardValidation' : ['prevalidation','validation','validationHarvesting'],
                   'standardValidationNoHLT' : ['prevalidationNoHLT','validationNoHLT','validationHarvestingNoHLT'],
                   'HGCalValidation' : ['', 'globalValidationHGCal', '']
                 }

_phase2_allowed = ['baseValidation','trackingValidation','muonOnlyValidation','JetMETOnlyValidation','bTagOnlyValidation','hcalOnlyValidation', 'HGCalValidation']
autoValidation['phase2Validation'] = ['','','']
for i in range(0,3):
    autoValidation['phase2Validation'][i] = '+'.join([autoValidation[m][i] for m in _phase2_allowed])
