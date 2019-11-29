import FWCore.ParameterSet.Config as cms

# magnetic field
# tracker geometry
# global tracker geometry
#from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
# tracker geometry
# tracker numbering
# ctfV0Producer
from RecoVertex.V0Producer.generalV0Candidates_cfi import *

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(generalV0Candidates, tkPtCut = 999.)
