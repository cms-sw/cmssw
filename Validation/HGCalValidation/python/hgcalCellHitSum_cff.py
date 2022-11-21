import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalCellHitSumEE_cfi import *

hgcalCellHitSumHEF = hgcalCellHitSumEE.clone(
    simhits = ('g4SimHits', 'HGCHitsHEfront'),
    detector = 'HGCalHESiliconSensitive',
)

hgcalCellHitSumHEB = hgcalCellHitSumEE.clone(
    simhits = ('g4SimHits', 'HGCHitsHEback'),
    detector   = 'HGCalHEScintillatorSensitive',
)
