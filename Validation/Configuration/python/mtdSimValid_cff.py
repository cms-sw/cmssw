import FWCore.ParameterSet.Config as cms

# MTD validation sequences
from Validation.MtdValidation.btlSimHitsValid_cfi import btlSimHitsValid
from Validation.MtdValidation.btlDigiHitsValid_cfi import btlDigiHitsValid
from Validation.MtdValidation.btlLocalRecoValid_cfi import btlLocalRecoValid
from Validation.MtdValidation.etlLocalRecoValid_cfi import etlLocalRecoValid
from Validation.MtdValidation.etlSimHitsValid_cfi import etlSimHitsValid
from Validation.MtdValidation.etlDigiHitsValid_cfi import etlDigiHitsValid
from Validation.MtdValidation.mtdTracksValid_cfi import mtdTracksValid
from Validation.MtdValidation.vertices4DValid_cfi import vertices4DValid
from Validation.MtdValidation.mtdEleIsoValid_cfi import mtdEleIsoValid

mtdSimValid  = cms.Sequence(btlSimHitsValid  + etlSimHitsValid )
mtdDigiValid = cms.Sequence(btlDigiHitsValid + etlDigiHitsValid)
mtdRecoValid = cms.Sequence(btlLocalRecoValid  + etlLocalRecoValid + mtdTracksValid + vertices4DValid + mtdEleIsoValid)

