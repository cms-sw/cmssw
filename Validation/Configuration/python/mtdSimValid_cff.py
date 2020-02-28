import FWCore.ParameterSet.Config as cms

# MTD validation sequences
from Validation.MtdValidation.btlSimHits_cfi import btlSimHits 
from Validation.MtdValidation.btlDigiHits_cfi import btlDigiHits 
from Validation.MtdValidation.btlRecHits_cfi import btlRecHits 
from Validation.MtdValidation.etlSimHits_cfi import etlSimHits
from Validation.MtdValidation.etlDigiHits_cfi import etlDigiHits
from Validation.MtdValidation.etlRecHits_cfi import etlRecHits

mtdSimValid  = cms.Sequence(btlSimHits  + etlSimHits )
mtdDigiValid = cms.Sequence(btlDigiHits + etlDigiHits)
mtdRecoValid = cms.Sequence(btlRecHits  + etlRecHits )
