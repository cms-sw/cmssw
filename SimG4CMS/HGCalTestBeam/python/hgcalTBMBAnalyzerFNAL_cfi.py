import FWCore.ParameterSet.Config as cms
from SimG4CMS.HGCalTestBeam.hgcalTBMBAnalyzer_cfi import *

hgcalTBMBAnalyzerFNAL = hgcalTBMBAnalyzer.clone(
    detectorNames = cms.vstring(
        'HGCCerenkov',
        'HGCMTS6SC1',
        'HGCTelescope',
        'HGCMTS6SC2',  
        'HGCMTS6SC3',  
        'HGCHeTube',
        'HGCFeChamber',
        'HGCScint1',
        'HGCScint2',
        'HGCFSiTrack',
        'HGCAlPlate',
        'HGCalExtra',
    ))
