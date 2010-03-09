# Fragment to switch off HCAL zero suppression as an option
# by cmsDriver customisation

# to generate Unsuppressed digis, one has to set the following parameter:
# process.simHcalDigis.useConfigZSvalues = 1
# to generate suppressed digis, useConfigZSvalues should be set to 0

import FWCore.ParameterSet.Config as cms
def customise(process):

#    process.hcalDigiSequence.replace(process.simHcalDigis,cms.SequencePlaceholder("simHcalDigis"))
#    process.load("SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisNoSuppression_cfi")

    process.simHcalDigis.HBlevel = -999
    process.simHcalDigis.HElevel = -999
    process.simHcalDigis.HOlevel = -999
    process.simHcalDigis.HFlevel = -999
    process.simHcalDigis.useConfigZSvalues = 1
    
    return(process)
