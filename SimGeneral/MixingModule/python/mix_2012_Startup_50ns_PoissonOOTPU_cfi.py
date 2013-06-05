import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.mixObjects_cfi import *
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *


mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(
      pixel = cms.PSet(
        pixelDigitizer
      ),
      strip = cms.PSet(
        stripDigitizer
      ),
      ecal = cms.PSet(
        ecalDigitizer
      ),
      hcal = cms.PSet(
        hcalDigitizer
      ),
      castor  = cms.PSet(
        castorDigitizer
      ),
      mergedtruth = cms.PSet(
        trackingParticles
      )
    ),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-2), ## in terms of 25 nsec

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59),
          probValue = cms.vdouble(
                                  2.344E-05,
                                  2.344E-05,
                                  2.344E-05,
                                  2.344E-05,
                                  4.687E-04,
                                  4.687E-04,
                                  7.032E-04,
                                  9.414E-04,
                                  1.234E-03,
                                  1.603E-03,
                                  2.464E-03,
                                  3.250E-03,
                                  5.021E-03,
                                  6.644E-03,
                                  8.502E-03,
                                  1.121E-02,
                                  1.518E-02,
                                  2.033E-02,
                                  2.608E-02,
                                  3.171E-02,
                                  3.667E-02,
                                  4.060E-02,
                                  4.338E-02,
                                  4.520E-02,
                                  4.641E-02,
                                  4.735E-02,
                                  4.816E-02,
                                  4.881E-02,
                                  4.917E-02,
                                  4.909E-02,
                                  4.842E-02,
                                  4.707E-02,
                                  4.501E-02,
                                  4.228E-02,
                                  3.896E-02,
                                  3.521E-02,
                                  3.118E-02,
                                  2.702E-02,
                                  2.287E-02,
                                  1.885E-02,
                                  1.508E-02,
                                  1.166E-02,
                                  8.673E-03,
                                  6.190E-03,
                                  4.222E-03,
                                  2.746E-03,
                                  1.698E-03,
                                  9.971E-04,
                                  5.549E-04,
                                  2.924E-04,
                                  1.457E-04,
                                  6.864E-05,
                                  3.054E-05,
                                  1.282E-05,
                                  5.081E-06,
                                  1.898E-06,
                                  6.688E-07,
                                  2.221E-07,
                                  6.947E-08,
                                  2.047E-08),
          histoFileName = cms.untracked.string('histProbFunction.root'),
        ),
	sequential = cms.untracked.bool(False),
        manage_OOT = cms.untracked.bool(True),  ## manage out-of-time pileup
        ## setting this to True means that the out-of-time pileup
        ## will have a different distribution than in-time, given
        ## by what is described on the next line:
        OOT_type = cms.untracked.string('Poisson'),  ## generate OOT with a Poisson matching the number chosen for in-time
        #OOT_type = cms.untracked.string('fixed'),  ## generate OOT with a fixed distribution
        #intFixed_OOT = cms.untracked.int32(2),
        fileNames = FileNames
    ),
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)



