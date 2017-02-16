import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.simHitValidation_cff    import *
from Validation.HGCalValidation.digiValidation_cff      import *
from Validation.HGCalValidation.recHitValidation_cff    import *
from Validation.HGCalValidation.hgcGeometryValidation_cfi import *

hgcalValidation = cms.Sequence(hgcGeomAnalysis+hgcalSimHitValidationEE+hgcalSimHitValidationHEF+hgcalSimHitValidationHEB+hgcalDigiValidationEE+hgcalDigiValidationHEF+hgcalDigiValidationHEB+hgcalRecHitValidationEE+hgcalRecHitValidationHEF+hgcalRecHitValidationHEB)
