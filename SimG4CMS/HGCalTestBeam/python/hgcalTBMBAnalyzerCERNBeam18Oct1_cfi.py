import FWCore.ParameterSet.Config as cms
from SimG4CMS.HGCalTestBeam.hgcalTBMBAnalyzer_cfi import *

hgcalTBMBAnalyzerCERNBeam18Oct1 = hgcalTBMBAnalyzer.clone(
    detectorNames = cms.vstring(
        'HGCalBeamWChamb',  
        'HGCalBeamS1',
        'HGCalBeamS2',
        'HGCalBeamS3',
        'HGCalBeamS4',
        'HGCalBeamS5',
        'HGCalBeamS6',
        'HGCalBeamCK3',
        'HGCalBeamHaloCounter',
        'HGCalBeamMuonCounter'
    ))
