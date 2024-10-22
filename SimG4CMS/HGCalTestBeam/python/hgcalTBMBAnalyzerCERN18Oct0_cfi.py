import FWCore.ParameterSet.Config as cms
from SimG4CMS.HGCalTestBeam.hgcalTBMBAnalyzer_cfi import *

hgcalTBMBAnalyzerCERN18Oct0 = hgcalTBMBAnalyzer.clone(
    detectorNames = cms.vstring(
        'HGCalBeamWChamb',  
        'HGCalBeamAl1',
        'HGCalBeamAl2',
        'HGCalBeamAl3',
        'HGCalBeamTube1',
        'HGCalBeamTube2',
        'HGCalBeamTube3',
        'HGCalEE',
        'HGCalHE',
        'HGCalAH'
    ))
