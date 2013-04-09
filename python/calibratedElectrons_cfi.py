
import FWCore.ParameterSet.Config as cms


#==============================================================================
# corrected pat electrons
#==============================================================================

calibratedElectrons = cms.EDProducer("CalibratedElectronProducer",

    # input collections
    inputElectronsTag = cms.InputTag('gsfElectrons'),
    # name of the ValueMaps containing the regression outputs                               
    nameEnergyReg = cms.InputTag('eleRegressionEnergy:eneRegForGsfEle'),
    nameEnergyErrorReg = cms.InputTag('eleRegressionEnergy:eneErrorRegForGsfEle'),
    # The rechits are needed to compute r9                                     
    recHitCollectionEB = cms.InputTag('reducedEcalRecHitsEB'),
    recHitCollectionEE = cms.InputTag('reducedEcalRecHitsEE'),

    outputGsfElectronCollectionLabel = cms.string('calibratedGsfElectrons'),
    # For conveniency  the ValueMaps are re-created with the new collection as key. The label of the ValueMap are defined below
    nameNewEnergyReg = cms.string('eneRegForGsfEle'),
    nameNewEnergyErrorReg  = cms.string('eneErrorRegForGsfEle'),                                     
                                         
    # data or MC corrections
    # if isMC is false, data corrections are applied
    isMC = cms.bool(False),
    
    # set to True to get more printout   
    verbose = cms.bool(False),

    # set to True to get special "fake" smearing for synchronization. Use JUST in case of synchronization
    synchronization = cms.bool(False),

    updateEnergyError = cms.bool(True),

    correctionsType = cms.int32(1),
    combinationType = cms.int32(1),
    
    lumiRatio = cms.double(0.0),
    # input datasets
    # Prompt means May10+Promptv4+Aug05+Promptv6 for 2011
    # ReReco means Jul05+Aug05+Oct03 for 2011
    # Jan16ReReco means Jan16 for 2011
    # Summer11 means summer11 MC..
    #inputDataset = cms.string("ReReco"),
    inputDataset = cms.string("Prompt")
    
)


