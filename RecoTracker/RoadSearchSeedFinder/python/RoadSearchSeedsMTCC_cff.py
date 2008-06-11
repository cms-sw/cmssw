import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsMTCCGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerMTCC_cff import *
import RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi
# RoadSearchSeedFinder
roadSearchSeedsMTCC = RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi.roadSearchSeeds.clone()
roadSearchSeedsMTCC.Mode = 'COSMICS'
roadSearchSeedsMTCC.RoadsLabel = 'MTCC'
roadSearchSeedsMTCC.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsMTCC.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsMTCC.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsMTCC.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsMTCC.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsMTCC.OuterSeedRecHitAccessUseStereo = True

