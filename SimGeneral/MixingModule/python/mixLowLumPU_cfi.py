# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the low-luminosity phase
mix = cms.EDFilter("MixingModule",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    Label = cms.string(''),
    bunchspace = cms.int32(25), ## nsec

    playback = cms.untracked.bool(False),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(2.8)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/mc/2007/12/17/RelVal-RelValMinBias-1197885154/0000/28A8801A-9BAC-DC11-8185-000423D992A4.root', '/store/mc/2007/12/17/RelVal-RelValMinBias-1197885154/0000/68BFD683-9DAC-DC11-BFEC-000423D6B5C4.root', '/store/mc/2007/12/17/RelVal-RelValMinBias-1197885154/0000/92F813F6-99AC-DC11-B88A-000423D993C0.root')
    )
)


