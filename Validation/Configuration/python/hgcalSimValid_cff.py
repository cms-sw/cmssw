import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.simHitValidationV6_cff    import *
from Validation.HGCalValidation.digiValidationV6_cff      import *
from Validation.HGCalValidation.recHitValidationV6_cff    import *
from Validation.HGCalValidation.hgcGeometryValidation_cfi import *

hgcalValidation = cms.Sequence(hgcGeomAnalysis+hgcalSimHitValidationEE+hgcalSimHitValidationHEF+hgcalSimHitValidationHEB+hgcalDigiValidationEE+hgcalDigiValidationHEF+hgcalDigiValidationHEB+hgcalRecHitValidationEE+hgcalRecHitValidationHEF+hgcalRecHitValidationHEB)
