import FWCore.ParameterSet.Config as cms
from SimG4CMS.HGCalTestBeam.hgcalTBMBAnalyzer_cfi import *

hgcalTBMBAnalyzerCERNBeam18Oct0 = hgcalTBMBAnalyzer.clone(
    detectorNames = cms.vstring(
        'HGCalBeamWChamb',  
        'HGCalBeamAl1',
        'HGCalBeamAl2',
        'HGCalBeamAl3',
        'HGCalBeamTube1',
        'HGCalBeamTube2',
        'HGCalBeamTube3',
        'HGCalMCPAluminium',
        'HGCalMCPPyrexGlass',
        'HGCalMCPLeadGlass'
    ))
