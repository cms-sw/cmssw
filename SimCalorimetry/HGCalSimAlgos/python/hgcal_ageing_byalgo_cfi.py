import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalSimAlgos.hgcSensorOpParams_cfi import *

def customise_hgcal_ageing_byalgo(process, **kwargs):
    """the following function configures the ageing conditions producer"""

    si_benchmark = kwargs.get('si_benchmark', 'TDR_600V')

    if not hasattr(process, 'ProcessAcceleratorCUDA'):
        process.load('Configuration.StandardSequences.Accelerators_cff')
        
    process.hgCalAlgoBasedAgeingCondProducer = cms.ESProducer(
        'hgcal::HGCalAlgoBasedAgeingCondProducer@alpaka',
        doseMapURL = cms.string(kwargs.get('doseMapURL', 'SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt')),
        doseMapAlgo = cms.uint32(kwargs.get('doseMapAlgo', 0)),
        scaleByDoseFactor = cms.double(kwargs.get('scaleByDoseFactor', 1.0)),
        si = cms.PSet(
            ileakParam = cms.vdouble(hgcSiSensorIleak(si_benchmark)),
            cceParams = cms.PSet(
                cceParam120 = cms.vdouble(hgcSiSensorCCE(120,si_benchmark)),
                cceParam200 = cms.vdouble(hgcSiSensorCCE(200,si_benchmark)),
                cceParam300 = cms.vdouble(hgcSiSensorCCE(300,si_benchmark))
            )
        ),
        sipm = cms.PSet(
            refIdark = cms.double(0.125),
            refXtalk = cms.double(0.01)
        )
    )

    return process

customise_hgcal_ageing_byalgo_startup = lambda process, **kwargs: customise_hgcal_ageing_byalgo(process, doseMapAlgo=3, **kwargs)
customise_hgcal_ageing_byalgo_eol = lambda process, **kwargs: customise_hgcal_ageing_byalgo(process, doseMapAlgo=0, **kwargs)
