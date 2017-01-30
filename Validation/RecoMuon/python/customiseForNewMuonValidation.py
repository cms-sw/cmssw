import FWCore.ParameterSet.Config as cms

def enableNewMuonVal(process):
    
    if ( hasattr(process,"validation") or  \
         hasattr(process,"globalValidation") ) and \
       hasattr(process,"recoMuonValidation") :

        print "[enableNewMuonVal] : pp RECO"

        process.load("Validation.RecoMuon.NewMuonValidation_cff")

        if hasattr(process,"globalValidation") :
            process.globalValidation.replace(process.recoMuonValidation, \
                                             process.NEWrecoMuonValidation)

        if hasattr(process,"validation") :
            process.validation.replace(process.recoMuonValidation, \
                                       process.NEWrecoMuonValidation)

        # CB I think probeTks need to accept loose

    if hasattr(process,"hltvalidation") and \
       hasattr(process,"recoMuonValidation") :

        print "[enableNewMuonVal] : HLT"

        process.load("Validation.RecoMuon.NewMuonValidationHLT_cff")

        if hasattr(process,"hltvalidation") :
            process.validation.replace(process.recoMuonValidationHLT_seq, \
                                       process.NEWrecoMuonValidationHLT_seq)


    if ( hasattr(process,"postValidation_preprod") or  \
         hasattr(process,"postValidation") )       and \
       hasattr(process,"recoMuonPostProcessors") :

        print "[enableNewMuonVal] : pp RECO Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidation.replace(process.recoMuonPostProcessors, \
                                           process.NEWrecoMuonPostProcessors)

        if hasattr(process,"postValidation_preprod") :
            process.postValidation_preprod.replace(process.recoMuonPostProcessors, \
                                                   process.NEWrecoMuonPostProcessors)


            
    if hasattr(process,"hltpostvalidation") and \
       hasattr(process,"recoMuonPostProcessorsHLT") :

        print "[enableNewMuonVal] : HLT Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessorHLT_cff")
        if hasattr(process,"hltpostvalidation") :
            process.hltpostvalidation.replace(process.recoMuonPostProcessorsHLT, \
                                              process.NEWrecoMuonPostProcessorsHLT)
    

    return process
    


    
