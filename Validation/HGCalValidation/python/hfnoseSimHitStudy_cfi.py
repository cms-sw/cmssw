import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitStudy_cfi import *

hgcalSimHitStudy.detectorNames  = ['HGCalHFNoseSensitive']
hgcalSimHitStudy.caloHitSources = ['HFNoseHits']
hgcalSimHitStudy.rMin    = 0
hgcalSimHitStudy.rMax    = 1500
hgcalSimHitStudy.zMin    = 10000
hgcalSimHitStudy.zMax    = 11000
hgcalSimHitStudy.etaMin  = 2.5
hgcalSimHitStudy.etaMax  = 5.5
hgcalSimHitStudy.nBinR   = 50
hgcalSimHitStudy.nBinZ   = 00
hgcalSimHitStudy.nBinEta = 50
hgcalSimHitStudy.ifNose  = True
hgcalSimHitStudy.layers  = 8
hgcalSimHitStudy.ifLayer = True
