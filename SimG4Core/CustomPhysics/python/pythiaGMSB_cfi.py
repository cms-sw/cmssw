# The following comments couldn't be translated into the new config version:

#       "MSTP(61)=0",   # no ISR
#       "MSTP(71)=0",   # no FSR
#       "MSTP(81)=0",   # no MPI
#       "MSTP(111)=0", # no hadronisation

# decay all unstable particles, regadless of their ctau
import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
source = cms.Source("PythiaSource",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    maxEvents = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.untracked.double(14000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=39                  ! All SUSY processes ', 
            'IMSS(1) = 11             ! Spectrum from external SLHA file', 
            'IMSS(21) = 33            ! LUN number for SLHA File (must be 33) ', 
            'IMSS(22) = 33', 
            'IMSS(11) = 1             ! gravitino as LSP ', 
            'RMSS(21) = 4000000       ! gravitino mass in eV'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'pythia', 
            'processParameters', 
            'SLHAParameters'),
        pythia = cms.vstring('MSTP(128) = 2', 
            'MSTJ(22) = 1'),
        SLHAParameters = cms.vstring('SLHAFILE = "isa-slha.out" ')
    )
)


