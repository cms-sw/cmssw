# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

#  this is the configuration to model pileup in StageA with 43x43 bunchcrossings
mix = cms.EDFilter("MixingModule",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 nsec

    Label = cms.string(''),
    bunchspace = cms.int32(1632), ## ns

    playback = cms.untracked.bool(False),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(0.0061)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/7A051E1C-4799-DC11-81FD-000423D99E46.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/9C659A18-4799-DC11-8899-000423D6C8E6.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/AADBC61C-4799-DC11-A162-000423D98AF0.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/B81B6DF7-2898-DC11-8ED5-001617E30F56.root')
    )
)


