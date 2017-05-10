import FWCore.ParameterSet.Config as cms

def enableNewMuonVal(process):
    "Enable new muon validation sequence, both for sources and harvesting"    
    
    if hasattr(process,"validation")       and \
       not hasattr(process,"validationHI") and \
       hasattr(process,"recoMuonValidation") :
    
        print "[enableNewMuonVal] : pp RECO"
    
        process.load("Validation.RecoMuon.NewMuonValidation_cff")
    
        if hasattr(process,"validation") :
            process.validation.replace(process.recoMuonValidation, \
                                       process.NEWrecoMuonValidation)

    if hasattr(process,"hltvalidation")    and \
       not hasattr(process,"validationHI") and \
       hasattr(process,"recoMuonValidation") :

        print "[enableNewMuonVal] : HLT"

        process.load("Validation.RecoMuon.NewMuonValidationHLT_cff")
        
        if hasattr(process,"hltvalidation") :
            process.hltvalidation.replace(process.recoMuonValidationHLT_seq, \
                                          process.NEWrecoMuonValidationHLT_seq)


    if hasattr(process,"globalValidationCosmics") and \
       hasattr(process,"recoCosmicMuonValidation") :

        print "[enableNewMuonVal] : Cosmic RECO"

        process.load("Validation.RecoMuon.NewMuonValidation_cff")

        if hasattr(process,"validationCosmics") :
            process.validation.replace(process.recoCosmicMuonValidation, \
                                       process.NEWrecoCosmicMuonValidation)


    if hasattr(process,"validation")              and \
       hasattr(process,"validationHI")            and \
       hasattr(process,"hiRecoMuonPrevalidation") and \
       hasattr(process,"hiRecoMuonValidation") :

        print "[enableNewMuonVal] : HI RECO"

        process.load("Validation.RecoHI.NewMuonValidationHeavyIons_cff")

        if hasattr(process,"prevalidation") :
            print "[enableNewMuonVal] : HI RECO prevalidation"
            process.prevalidation.replace(process.hiRecoMuonPrevalidation, \
                                          process.NEWhiRecoMuonPrevalidation)

        if hasattr(process,"validation") :
            print "[enableNewMuonVal] : HI RECO validation"
            process.validation.replace(process.hiRecoMuonValidation, \
                                       process.NEWhiRecoMuonValidation)
        
    if hasattr(process,"postValidation") and \
       hasattr(process,"recoMuonPostProcessors") :

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidation.replace(process.recoMuonPostProcessors, \
                                           process.NEWrecoMuonPostProcessors)

    if hasattr(process,"postValidationHI") and \
       hasattr(process,"recoMuonPostProcessors") :

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidationHI.replace(process.recoMuonPostProcessors, \
                                             process.NEWrecoMuonPostProcessors)


    if hasattr(process,"postValidation_fastsim") and \
       hasattr(process,"recoMuonPostProcessors") :

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation_fastsim") :
            process.postValidation_fastsim.replace(process.recoMuonPostProcessors, \
                                                   process.NEWrecoMuonPostProcessors)

            
    if hasattr(process,"hltpostvalidation") and \
       hasattr(process,"recoMuonPostProcessorsHLT") :

        process.load("Validation.RecoMuon.NewPostProcessorHLT_cff")
        if hasattr(process,"hltpostvalidation") :
            process.hltpostvalidation.replace(process.recoMuonPostProcessorsHLT, \
                                              process.NEWrecoMuonPostProcessorsHLT)
    

    if hasattr(process,"postValidationCosmics") and \
       hasattr(process,"postProcessorMuonMultiTrack") :

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidationCosmics") :
            process.postValidationCosmics.replace(process.postProcessorMuonMultiTrack, \
                                                  process.NEWpostProcessorMuonTrack)

    return process
