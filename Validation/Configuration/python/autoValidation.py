autoValidation = { 'liteTracking' : ['prevalidationLiteTracking','validationLiteTracking','validationHarvesting'],
                   'trackingOnlyValidation' : ['globalPrevalidationTrackingOnly','globalValidationTrackingOnly','postValidation_trackingOnly'],
                   'muonOnlyValidation' : ['globalPrevalidationMuons','globalValidationMuons','postValidation_muons'],
                   'bTagOnlyValidation' : ['prebTagSequenceMC','bTagPlotsMCbcl','bTagCollectorSequenceMCbcl'],
                   'JetMETOnlyValidation' : ['globalPrevalidationJetMETOnly','globalValidationJetMETonly','postValidation_JetMET'],
                   'baseValidation' : ['baseCommonPreValidation','baseCommonValidation','postValidation_common'],
                   'miniAODValidation' : ['prevalidationMiniAOD','validationMiniAOD','validationHarvestingMiniAOD'],
                   'standardValidation' : ['prevalidation','validation','validationHarvesting']
                 }

_phase2_allowed = ['baseValidation','trackingOnlyValidation','muonOnlyValidation','JetMETOnlyValidation','bTagOnlyValidation']
autoValidation['phase2Validation'] = ['','','']
for i in range(0,3):
    autoValidation['phase2Validation'][i] = '+'.join([autoValidation[m][i] for m in _phase2_allowed])
