import FWCore.ParameterSet.Config as cms

# reproduce JetCorrector modules when not available
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrector_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrector_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrector_cfi import *

hltAK4PFPuppiJetCorrectorL1_ForValidation = cms.EDProducer("L1FastjetCorrectorProducer", 
    algorithm = cms.string('AK4PFPuppiHLT'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("hltFixedGridRhoFastjetAll"),
    skipMissingProduct = cms.bool(True)
)
hltAK4PFPuppiJetCorrectorL2_ForValidation = hltAK4PFPuppiJetCorrectorL2.clone()
hltAK4PFPuppiJetCorrectorL3_ForValidation = hltAK4PFPuppiJetCorrectorL3.clone()
hltAK4PFPuppiJetCorrector_ForValidation = hltAK4PFPuppiJetCorrector.clone(
    correctors = cms.VInputTag(
        "hltAK4PFPuppiJetCorrectorL1_ForValidation",
        "hltAK4PFPuppiJetCorrectorL2_ForValidation", 
        "hltAK4PFPuppiJetCorrectorL3_ForValidation")
)

hltAK4PFJetCorrectorL1_ForValidation = cms.EDProducer("L1FastjetCorrectorProducer", 
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("hltFixedGridRhoFastjetAll"),
    skipMissingProduct = cms.bool(True)
)
hltAK4PFJetCorrectorL2_ForValidation = hltAK4PFJetCorrectorL2.clone()
hltAK4PFJetCorrectorL3_ForValidation = hltAK4PFJetCorrectorL3.clone()
hltAK4PFJetCorrector_ForValidation = hltAK4PFJetCorrector.clone(
    correctors = cms.VInputTag(
        "hltAK4PFJetCorrectorL1_ForValidation",
        "hltAK4PFJetCorrectorL2_ForValidation", 
        "hltAK4PFJetCorrectorL3_ForValidation")
)

hltAK4PFCHSJetCorrectorL1_ForValidation = cms.EDProducer("L1FastjetCorrectorProducer", 
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("hltFixedGridRhoFastjetAll"),
    skipMissingProduct = cms.bool(True)
)
hltAK4PFCHSJetCorrectorL2_ForValidation = hltAK4PFCHSJetCorrectorL2.clone()
hltAK4PFCHSJetCorrectorL3_ForValidation = hltAK4PFCHSJetCorrectorL3.clone()
hltAK4PFCHSJetCorrector_ForValidation = hltAK4PFCHSJetCorrector.clone(
    correctors = cms.VInputTag(
        "hltAK4PFCHSJetCorrectorL1_ForValidation",
        "hltAK4PFCHSJetCorrectorL2_ForValidation", 
        "hltAK4PFCHSJetCorrectorL3_ForValidation")
)

hltJetCorrectionTask = cms.Task(
    hltAK4PFPuppiJetCorrectorL1_ForValidation,
    hltAK4PFPuppiJetCorrectorL2_ForValidation,
    hltAK4PFPuppiJetCorrectorL3_ForValidation,
    hltAK4PFPuppiJetCorrector_ForValidation,
    hltAK4PFJetCorrectorL1_ForValidation,
    hltAK4PFJetCorrectorL2_ForValidation,
    hltAK4PFJetCorrectorL3_ForValidation,
    hltAK4PFJetCorrector_ForValidation,
    hltAK4PFCHSJetCorrectorL1_ForValidation,
    hltAK4PFCHSJetCorrectorL2_ForValidation,
    hltAK4PFCHSJetCorrectorL3_ForValidation,
    hltAK4PFCHSJetCorrector_ForValidation,
)
