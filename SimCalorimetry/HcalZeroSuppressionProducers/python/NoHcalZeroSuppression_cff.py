# Fragment to switch off HCAL zero suppression as an option
# by cmsDriver customisation

import FWCore.ParameterSet.Config as cms
def customise(process):

#    process.hcalDigiSequence.replace(process.simHcalDigis,cms.SequencePlaceholder("simHcalDigis"))
#    process.load("SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisNoSuppression_cfi")

    process.simHcalDigis.HBlevel = -999
    process.simHcalDigis.HElevel = -999
    process.simHcalDigis.HOlevel = -999
    process.simHcalDigis.HFlevel = -999
    
    return(process)
