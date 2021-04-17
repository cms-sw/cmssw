import FWCore.ParameterSet.Config as cms

# magnetic field
# tracker geometry
# global tracker geometry
#from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
# tracker geometry
# tracker numbering
# ctfV0Producer
from RecoVertex.V0Producer.generalV0Candidates_cfi import *

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(generalV0Candidates, tkPtCut = 999.)
