import FWCore.ParameterSet.Config as cms

#--- reset HB/HE ZS to 2TS 
#--- NB: may need appropriate HcalZSThresholds update
  
def customise_2TS(process):
    process.simHcalDigis.HBregion = (2,5)
    process.simHcalDigis.HEregion = (2,5)
    process.simHcalDigis.use1ts = False
    return(process)

