import FWCore.ParameterSet.Config as cms

def enableNewMuonVal(process):
    "Enable new muon validation sequence, both for sources and harvesting"    
    
    #CB I need to find a better IF statement here
    if hasattr(process,"validation") and \
       not hasattr(process,"validationHI") and \
       hasattr(process,"globalValidation") and \
       hasattr(process,"recoMuonValidation") :
    
        print "[enableNewMuonVal] : pp RECO"
    
        process.probeTracks.quality = cms.vstring('loose')
    
        process.load("Validation.RecoMuon.NewMuonValidation_cff")
    
        if hasattr(process,"validation") :
            print "[enableNewMuonVal] : pp RECO validation"
            process.validation.replace(process.recoMuonValidation, \
                                       process.NEWrecoMuonValidation)
            print process.validation      


    #CB I need to find a better IF statement here
    if hasattr(process,"hltvalidation") and \
       not hasattr(process,"validationHI") and \
       hasattr(process,"recoMuonValidation") :

        print "[enableNewMuonVal] : HLT"

        process.load("Validation.RecoMuon.NewMuonValidationHLT_cff")
        
        if hasattr(process,"hltvalidation") :
            process.validation.replace(process.recoMuonValidationHLT_seq, \
                                       process.NEWrecoMuonValidationHLT_seq)


    if hasattr(process,"globalValidationCosmics") and \
       hasattr(process,"recoMuonValidationCosmics") :

        print "[enableNewMuonVal] : Cosmic RECO"

        probeTracks.quality = cms.vstring('loose')

        process.load("Validation.RecoMuon.NewMuonValidation_cff")

        if hasattr(process,"validationCosmics") :
            process.validation.replace(process.recoMuonValidationCosmics, \
                                       process.NEWrecoMuonValidationCosmics)


    if hasattr(process,"validation") and \
       hasattr(process,"validationHI") and \
       hasattr(process,"hiRecoMuonPrevalidation") and \
       hasattr(process,"hiRecoMuonValidation") :

        print "[enableNewMuonVal] : HI RECO"

        process.probeTracks.quality = cms.vstring('loose')

        process.load("Validation.RecoHI.NewMuonValidationHeavyIons_cff")

        if hasattr(process,"prevalidation") :
            print "[enableNewMuonVal] : HI RECO validation preVal"
            process.prevalidation.replace(process.hiRecoMuonPrevalidation, \
                                          process.NEWhiRecoMuonPrevalidation)

        if hasattr(process,"validation") :
            print "[enableNewMuonVal] : HI RECO validation globalVal"
            process.validation.replace(process.hiRecoMuonValidation, \
                                       process.NEWhiRecoMuonValidation)
        
    if hasattr(process,"postValidation") and \
       hasattr(process,"recoMuonPostProcessors") :

        print "[enableNewMuonVal] : pp RECO Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidation.replace(process.recoMuonPostProcessors, \
                                           process.NEWrecoMuonPostProcessors)

    if hasattr(process,"postValidationHI") and \
       hasattr(process,"recoMuonPostProcessors") :

        print "[enableNewMuonVal] : pp RECO Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation") :
            process.postValidationHI.replace(process.recoMuonPostProcessors, \
                                             process.NEWrecoMuonPostProcessors)


    if hasattr(process,"postValidation_fastsim") and \
       hasattr(process,"recoMuonPostProcessors") :

        print "[enableNewMuonVal] : pp RECO Harvesting (FastSim)"

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidation_fastsim") :
            process.postValidation_fastsim.replace(process.recoMuonPostProcessors, \
                                                   process.NEWrecoMuonPostProcessors)

            
    if hasattr(process,"hltpostvalidation") and \
       hasattr(process,"recoMuonPostProcessorsHLT") :

        print "[enableNewMuonVal] : HLT Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessorHLT_cff")
        if hasattr(process,"hltpostvalidation") :
            process.hltpostvalidation.replace(process.recoMuonPostProcessorsHLT, \
                                              process.NEWrecoMuonPostProcessorsHLT)
    

    if hasattr(process,"postValidationCosmics") and \
       hasattr(process,"postProcessorMuonMultiTrack") :

        print "[enableNewMuonVal] : pp RECO Harvesting"

        process.load("Validation.RecoMuon.NewPostProcessor_cff")

        if hasattr(process,"postValidationCosmics") :
            process.postValidationCosmics.replace(process.postProcessorMuonMultiTrack, \
                                           process.NEWpostProcessorMuonTrack)

    return process

    



def runMuonValForTesting(process):
    "Customise step3 to run only muon validation (expects both step2 and step 3 files in input)"

    from Validation.RecoMuon.muonValidation_cff import *

    muonValidation_seq.remove(trackAssociatorByHits)
    muonValidation_seq.remove(tpToTkmuTrackAssociation)
    muonValidation_seq.remove(trkMuonTrackVTrackAssoc)
    #process.muonValidation_seq.remove(process.muonValidationRMV_seq)

    process.recoMuonValidationForTesting_seq = cms.Sequence( muonValidation_seq 
                                                             + muonValidationTEV_seq 
                                                             + muonValidationRefit_seq 
                                                             )

    process.validationForTesting = cms.Sequence( cms.SequencePlaceholder("mix") 
                                                 + process.recoMuonValidationForTesting_seq 
                                                 + process.recoMuonValidationHLT_seq
                                               )

    process.outputForTesting = cms.OutputModule("PoolOutputModule",
                                                fileName = cms.untracked.string('events.root'),
                                                splitLevel = cms.untracked.int32(0)
                                                )

    if hasattr(process,"validation_step") :
        print "[runMuonValForTesting] : customising validation"
        process.validation_step = cms.Path(process.validationForTesting)

    # CB what to do with this?
    #if hasattr(process,"output_step") :
    #    print "[runMuonValForTesting] : customising output"
    #    process.output_step = cms.EndPath(process.outputForTesting)

    process.schedule = cms.Schedule(process.validation_step,process.DQMoutput_step)

    return process
    



def runMuonHarvestingForTesting(process):
    "Customise harvesting to run only muon validation"

    print "[runMuonValForTesting] : customising harvesting"

    process.validationHarvestingForTesting = cms.Path(process.recoMuonPostProcessors
                                                      + process.recoMuonPostProcessorsHLT)

    process.schedule = cms.Schedule(process.validationHarvestingForTesting,process.dqmsave_step)

    return process
    
