import FWCore.ParameterSet.Config as cms

from Validation.HcalDigis.HcalDigisParam_cfi import *

hltHCALdigisAnalyzer = hcaldigisAnalyzer.clone()
hltHCALdigisAnalyzer.dirName      = cms.untracked.string("HLT/HCAL/Digis/Simulation")
hltHCALdigisAnalyzer.digiTag	  = cms.InputTag("hcalDigis")    
hltHCALdigisAnalyzer.QIE10digiTag = cms.InputTag("hcalDigis")    
hltHCALdigisAnalyzer.QIE11digiTag = cms.InputTag("hcalDigis")
hltHCALdigisAnalyzer.mode	  = cms.untracked.string('multi')
hltHCALdigisAnalyzer.hcalselector = cms.untracked.string('all')    
hltHCALdigisAnalyzer.mc		  = cms.untracked.string('yes')    
hltHCALdigisAnalyzer.simHits      = cms.untracked.InputTag("g4SimHits","HcalHits")    
hltHCALdigisAnalyzer.emulTPs      = cms.InputTag("emulDigis")    
hltHCALdigisAnalyzer.dataTPs      = cms.InputTag("simHcalTriggerPrimitiveDigis")    
hltHCALdigisAnalyzer.TestNumber   = cms.bool(False)    
hltHCALdigisAnalyzer.hep17        = cms.bool(False)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hltHCALdigisAnalyzer,
    dataTPs = "DMHcalTriggerPrimitiveDigis",
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
from Configuration.Eras.Modifier_fastSim_cff import fastSim
(run2_HCAL_2017 & ~fastSim).toModify(hltHCALdigisAnalyzer,
    TestNumber    = cms.bool(True)
)

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hltHCALdigisAnalyzer,
    hep17 = cms.bool(True)
)

