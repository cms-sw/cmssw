import FWCore.ParameterSet.Config as cms

def enableNewMuonVal(process):
    
    if ( hasattr(process,"validation") or  \
         hasattr(process,"globalValidation") ) and \
       hasattr(process,"recoMuonValidation") :

        process.load("Validation.RecoMuon.NewMuonValidation_cff")

        if hasattr(process,"globalValidation") :
            process.globalValidation.replace(process.recoMuonValidation, \
                                             process.NEWrecoMuonValidation)

        if hasattr(process,"validation") :
            process.validation.replace(process.recoMuonValidation, \
                                       process.NEWrecoMuonValidation)

        # CB I think probeTks need to accept loose

    if ( hasattr(process,"postValidation_preprod") or  \
         hasattr(process,"postValidation") )       and \
       hasattr(process,"recoMuonPostProcessors") :

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidation.replace(process.recoMuonPostProcessors, \
                                           process.NEWrecoMuonPostProcessors)

        if hasattr(process,"postValidation_preprod") :
            process.postValidation_preprod.replace(process.recoMuonPostProcessors, \
                                                   process.NEWrecoMuonPostProcessors)
    
    return process
    
    
