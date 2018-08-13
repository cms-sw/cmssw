from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# here we define part of the configuration of MixingModule
# the rest, notably:
#            "input","bunchspace","minBunch","maxBunch"
# is to be retrieved from database
# see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRunDependentMC

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects 
from SimGeneral.MixingModule.mixPoolSource_cfi import * 
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",

    # this is where you activate reading from DB of: "input","bunchspace","minBunch","maxBunch"
    readDB = cms.bool(True),

    digitizers = cms.PSet(theDigitizers),                 
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(314159),    ## these three parameters are needed at instantiation time, BUT the actual value will NOT be used
    minBunch = cms.int32(-314159),   ## actual values will be retrieved from database

    bunchspace = cms.int32(314159),  ## [ditto] 
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('readDB'),
        sequential = cms.untracked.bool(False),                          
        fileNames = FileNames 
    ),

    mixObjects = cms.PSet(theMixObjects)                 
    #mixObjects = cms.PSet(
    #    mixCH = cms.PSet(
    #        mixCaloHits
    #    ),
    #    mixTracks = cms.PSet(
    #        mixSimTracks
    #    ),
    #    mixVertices = cms.PSet(
    #        mixSimVertices
    #    ),
    #    mixSH = cms.PSet(
    #        mixSimHits
    #    ),
    #    mixHepMC = cms.PSet(
    #        mixHepMCProducts
    #    )
    #)
)



if mix.readDB == cms.bool(True):
    print(' ')
    print('MixingModule will be configured from db; this is mix.readDB : ',mix.readDB)
else :
    print(' ')
    print('MixingModule is NOT going to be configured from db; this is mix.readDB : ',mix.readDB)
