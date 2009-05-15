import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from Configuration.StandardSequences.RawToDigi_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.StandardSequences.Geometry_cff import *

makeTausFromDigis = cms.Sequence(RawToDigi*reconstruction)
