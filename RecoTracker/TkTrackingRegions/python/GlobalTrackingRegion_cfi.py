import FWCore.ParameterSet.Config as cms

# deprecated.  Changing over to the block
# NO: must be completely removed, as the CFI is included at top level
# otherwise we get duplicate definitions
# PSet RegionPSet = {
#   double ptMin = 0.9
#   double originRadius = 0.2
#   double originHalfLength = 15.9
#   double originXPos = 0.0
#   double originYPos = 0.0
#   double originZPos = 0.0
#   bool precise = true
# }
RegionPSetBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originHalfLength = cms.double(21.2),
        originRadius = cms.double(0.2),
        originYPos = cms.double(0.0),
        ptMin = cms.double(0.9),
        originXPos = cms.double(0.0),
        originZPos = cms.double(0.0)
    )
)

