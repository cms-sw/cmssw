# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup for final scheme 
mix = cms.EDFilter("MixingModule",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    Label = cms.string(''),
    bunchspace = cms.int32(25), ## ns

    playback = cms.untracked.bool(False),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(10.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/140FBA0E-1599-DC11-B571-000423D99F3E.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/7A051E1C-4799-DC11-81FD-000423D99E46.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/9C659A18-4799-DC11-8899-000423D6C8E6.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/AADBC61C-4799-DC11-A162-000423D98AF0.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/B81B6DF7-2898-DC11-8ED5-001617E30F56.root')
    )
)


