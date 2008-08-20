import FWCore.ParameterSet.Config as cms

#
# tqaf layer1 event content is equivalent to pat layer0 & 1
#
# the EventContent might still be modified by including the
# files: * tqafLayer1_genParticles_cff
#        * tqafLayer1_jetCollections_cff
#
# in the process file and calling the corresponding macros
from PhysicsTools.PatAlgos.patLayer1_EventContent_cff import *

