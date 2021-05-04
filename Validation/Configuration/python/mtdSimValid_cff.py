import FWCore.ParameterSet.Config as cms

# MTD validation sequences
from Validation.MtdValidation.btlSimHits_cfi import btlSimHits 
from Validation.MtdValidation.btlDigiHits_cfi import btlDigiHits 
from Validation.MtdValidation.btlLocalReco_cfi import btlLocalReco
from Validation.MtdValidation.etlLocalReco_cfi import etlLocalReco
from Validation.MtdValidation.etlSimHits_cfi import etlSimHits
from Validation.MtdValidation.etlDigiHits_cfi import etlDigiHits
from Validation.MtdValidation.mtdTracks_cfi import mtdTracks

mtdSimValid  = cms.Sequence(btlSimHits  + etlSimHits )
mtdDigiValid = cms.Sequence(btlDigiHits + etlDigiHits)
mtdRecoValid = cms.Sequence(btlLocalReco  + etlLocalReco + mtdTracks)
