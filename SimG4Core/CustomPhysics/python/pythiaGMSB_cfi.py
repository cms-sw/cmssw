# The following comments couldn't be translated into the new config version:

# This is a vector of ParameterSet names to be read, in this order

#       "MSTP(61)=0",   # no ISR
#       "MSTP(71)=0",   # no FSR
#       "MSTP(81)=0",   # no MPI
#       "MSTP(111)=0", # no hadronisation

# decay all unstable particles, regadless of their ctau
import FWCore.ParameterSet.Config as cms

source = cms.Source("PythiaSource",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    maxEvents = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.untracked.double(14000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('MSEL=39                  ! All SUSY processes ', 'IMSS(1) = 11             ! Spectrum from external SLHA file', 'IMSS(21) = 33            ! LUN number for SLHA File (must be 33) ', 'IMSS(22) = 33', 'IMSS(11) = 1             ! gravitino as LSP ', 'RMSS(21) = 4000000       ! gravitino mass in eV'),
        parameterSets = cms.vstring('pythiaUESettings', 'pythia', 'processParameters', 'SLHAParameters'),
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 'MSTJ(22)=2     ! Decay those unstable particles', 'PARJ(71)=10 .  ! for which ctau  10 mm', 'MSTP(2)=1      ! which order running alphaS', 'MSTP(33)=0     ! no K factors in hard cross sections', 'MSTP(51)=7     ! structure function chosen', 'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 'MSTP(82)=4     ! Defines the multi-parton model', 'MSTU(21)=1     ! Check on possible errors during program execution', 'PARP(82)=1.9409   ! pt cutoff for multiparton interactions', 'PARP(89)=1960. ! sqrts for which PARP82 is set', 'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 'PARP(90)=0.16  ! Multiple interactions: rescaling power', 'PARP(67)=2.5    ! amount of initial-state radiation', 'PARP(85)=1.0  ! gluon prod. mechanism in MI', 'PARP(86)=1.0  ! gluon prod. mechanism in MI', 'PARP(62)=1.25   ! ', 'PARP(64)=0.2    ! ', 'MSTP(91)=1     !', 'PARP(91)=2.1   ! kt distribution', 'PARP(93)=15.0  ! '),
        SLHAParameters = cms.vstring('SLHAFILE = "isa-slha.out" '),
        pythia = cms.vstring('MSTP(128) = 2', 'MSTJ(22) = 1')
    )
)


