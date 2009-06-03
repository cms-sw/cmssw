import FWCore.ParameterSet.Config as cms

def harvest_only(process):
    process.validationHarvesting.remove(process.hltpostvalidation)

def validation_only(process):
    process.trackMCMatchSequence.remove(process.assoc2secStepTk)
    process.trackMCMatchSequence.remove(process.assoc2thStepTk)
    process.trackMCMatchSequence.remove(process.assoc2GsfTracks)
    process.only_validation_and_TP = cms.Sequence(process.mix
                                                  *process.trackingParticles
                                                  *process.tracksValidation
                                                  *process.recoMuonValidation
                                                  *process.HLTMuonVal
                                                  )
    process.validation_step.replace(process.validation,process.only_validation_and_TP)
    
