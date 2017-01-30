import FWCore.ParameterSet.Config as cms

def enableNewMuonVal(process):
    "Enable new muon validation sequence, both in step3 and harvesting"    
    
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

    probeTracks.quality = cms.vstring('loose')

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
    
