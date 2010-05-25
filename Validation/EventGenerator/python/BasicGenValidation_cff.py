import FWCore.ParameterSet.Config as cms

# Basic HepMC/GenParticle/GenJet validation modules
from Validation.EventGenerator.BasicHepMCValidation_cfi import *
from Validation.EventGenerator.BasicGenParticleValidation_cfi import *

# Analyzer for MB/UE studies
from Validation.EventGenerator.MBUEandQCDValidation_cff import *

# Duplication Checker, for LHE workflows
from Validation.EventGenerator.DuplicationChecker_cfi import *

# simple analyzer for DrellYan->lepon processes
from Validation.EventGenerator.DrellYanValidation_cfi import *

# define seqeunces...
basicGenTest_seq = cms.Sequence(basicHepMCValidation+basicGenParticleValidation)
duplicationChecker_seq = cms.Sequence(duplicationChecker)
mbueAndqcdValidation_seq = cms.Sequence(mbueAndqcd_seq)
drellYanValidation_seq = cms.Sequence(drellYanValidation)

genvalid = cms.Sequence(basicGenTest_seq)
genvalid_qcd = cms.Sequence(basicGenTest_seq+mbueAndqcdValidation_seq)
