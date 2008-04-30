import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# tracker geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# global tracker geometry
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# ctfV0Producer
from RecoVertex.V0Producer.generalV0Candidates_cfi import *

