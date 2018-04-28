import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hcaldigisAnalyzer = DQMEDAnalyzer('HcalDigisValidation',
    outputFile	= cms.untracked.string(''),
    digiTag	= cms.InputTag("hcalDigis"),
    QIE10digiTag= cms.InputTag("hcalDigis"),
    QIE11digiTag= cms.InputTag("hcalDigis"),
    mode	= cms.untracked.string('multi'),
    hcalselector= cms.untracked.string('all'),
    mc		= cms.untracked.string('yes'),
    simHits     = cms.untracked.InputTag("g4SimHits","HcalHits"),
    emulTPs     = cms.InputTag("emulDigis"),
    dataTPs     = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    TestNumber  = cms.bool(False),
    hep17       = cms.bool(False),
    HEPhase1  	= cms.bool(False),
    HBPhase1	= cms.bool(False),
    Plot_TP_ver = cms.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(hcaldigisAnalyzer, simHits = "fastSimProducer:HcalHits")

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hcaldigisAnalyzer,
    dataTPs = "DMHcalTriggerPrimitiveDigis",
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
(run2_HCAL_2017 & ~fastSim).toModify(hcaldigisAnalyzer,
    TestNumber    = cms.bool(True)
)

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hcaldigisAnalyzer,
    hep17 = cms.bool(True)
)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify(hcaldigisAnalyzer,
    HEPhase1 = cms.bool(True)
)
    
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(hcaldigisAnalyzer,
    HBPhase1 = cms.bool(True)
)
