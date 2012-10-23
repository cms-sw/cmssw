
import FWCore.ParameterSet.Config as cms


#==============================================================================
# corrected pat electrons
#==============================================================================

calibratedElectrons = cms.EDProducer("CalibratedElectronProducer",

    # input collections
    inputElectronsTag = cms.InputTag('gsfElectrons'),
    nameEnergyReg = cms.InputTag('eleRegressionEnergy:eneRegForGsfEle'),
    nameEnergyErrorReg = cms.InputTag('eleRegressionEnergy:eneErrorRegForGsfEle'),
    recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
    recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE'),

    outputGsfElectronCollectionLabel = cms.string('calibratedGsfElectrons'),
# For conveniency                                     
    nameNewEnergyReg = cms.string('eneRegForGsfEle'),
    nameNewEnergyErrorReg  = cms.string('eneErrorRegForGsfEle'),                                     
                                         
    # data or MC corrections
    # if isMC is false, data corrections are applied
    isMC = cms.bool(False),

    # set to True to read AOD format
    isAOD = cms.bool(False),
    
    # set to True to get debugging printout   
    debug = cms.bool(False),
 
    updateEnergyError = cms.bool(True),

    #set to 0 to not apply corrections
    #set to 1 to apply regression_1 corrections
    #set to 2 to apply regression_2 corrections
    #set to 999 to apply default corrections
    applyCorrections = cms.int32(999),
    
    # input datasets
    # Prompt means May10+Promptv4+Aug05+Promptv6 for 2011
    # ReReco means Jul05+Aug05+Oct03 for 2011
    # Jan16ReReco means Jan16 for 2011
    # Summer11 means summer11 MC..
    #inputDataset = cms.string("ReReco"),
    inputDataset = cms.string("Prompt")
    
)


