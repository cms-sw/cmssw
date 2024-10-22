import FWCore.ParameterSet.Config as cms

from SimG4Core.PrintGeomInfo.printGeomSolids_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(printGeomSolids,
                fromDD4hep = cms.bool(True),
)
