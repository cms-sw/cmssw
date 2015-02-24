import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock

pixelDigitizer = cms.PSet(
    SiPixelSimBlock,
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
)


##
## Make changes for running in the Run 2 scenario
##
from Configuration.StandardSequences.Eras import eras

def modifyPixelDigitizerForRun2Bunchspacing25( digitizer ) :
    """
    Function that modifies the pixel digitiser for Run 2 with 25ns bunchspacing.
    First argument is the pixelDigitizer object (generally
    "process.mix.digitizers.pixel" when this function is called).
    """
    # DynamicInefficency - 13TeV - 25ns case
    digitizer.theInstLumiScaleFactor = cms.double(364)
    digitizer.theLadderEfficiency_BPix1 = cms.vdouble( [1]*20 ) # this syntax makes an array with 20 copies of "1"
    digitizer.theLadderEfficiency_BPix2 = cms.vdouble( [1]*32 )
    digitizer.theLadderEfficiency_BPix3 = cms.vdouble( [1]*44 )
    digitizer.theModuleEfficiency_BPix1 = cms.vdouble( 1, 1, 1, 1, )
    digitizer.theModuleEfficiency_BPix2 = cms.vdouble( 1, 1, 1, 1, )
    digitizer.theModuleEfficiency_BPix3 = cms.vdouble( 1, 1, 1, 1 )
    digitizer.thePUEfficiency_BPix1 = cms.vdouble( 1.00023, -3.18350e-06, 5.08503e-10, -6.79785e-14 )
    digitizer.thePUEfficiency_BPix2 = cms.vdouble( 9.99974e-01, -8.91313e-07, 5.29196e-12, -2.28725e-15 )
    digitizer.thePUEfficiency_BPix3 = cms.vdouble( 1.00005, -6.59249e-07, 2.75277e-11, -1.62683e-15 )

def modifyPixelDigitizerForRun2Bunchspacing50( digitizer ) :
    """
    Function that modifies the pixel digitiser for Run 2 with 50ns bunchspacing.
    
    First argument is the pixelDigitizer object (generally
    "process.mix.digitizers.pixel" when this function is called).
    """
    # DynamicInefficency - 13TeV - 50ns case
    digitizer.theInstLumiScaleFactor = cms.double(246.4)
    digitizer.theLadderEfficiency_BPix1 = cms.vdouble( [0.979259,0.976677]*10 ) # This syntax makes a 20 element array of alternating numbers
    digitizer.theLadderEfficiency_BPix2 = cms.vdouble( [0.994321,0.993944]*16 )
    digitizer.theLadderEfficiency_BPix3 = cms.vdouble( [0.996787,0.996945]*22 )

def _modifyPixelDigitizerForRun2( digitiserInstance ) :
    """
    Either delegates to modifyPixelDigitizerForRun2Bunchspacing25 or
    modifyPixelDigitizerForRun2Bunchspacing50 depending on which bunch
    spacing era is active.
    Can't simply use the bunchspacing era because I only want it enacted
    if this is also Run 2 (i.e. eras.run2 AND eras.bunchspacingXns active). 
    """
    if eras.bunchspacing25ns.isChosen() and eras.bunchspacing50ns.isChosen() :
        raise Exception( "ERROR: both the 25ns and 50ns bunch crossing eras are active. Take one of them out of the cms.Process constructor.")

    if eras.bunchspacing25ns.isChosen() :
        modifyPixelDigitizerForRun2Bunchspacing25( digitiserInstance )
    elif eras.bunchspacing50ns.isChosen() :
        modifyPixelDigitizerForRun2Bunchspacing50( digitiserInstance )

eras.run2_common.toModify( pixelDigitizer, func=_modifyPixelDigitizerForRun2 )

