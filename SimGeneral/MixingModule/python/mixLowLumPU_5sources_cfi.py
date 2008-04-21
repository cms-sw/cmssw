# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the low-luminosity phase
# here we have an example with 4 input sources
# but you are free to put only those you need
# or you can replace the type by "none" for a source you dont want
# please note that the names of the input sources are fixed: 'input', 'cosmics', 'beamhalo_minus', 'beamhalo_plus'
#
mix = cms.EDFilter("MixingModule",
    beamhalo_minus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/28FE7F00-43D3-DC11-8217-000423D98EC4.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/B0D04F90-4CD3-DC11-9B60-001617E30D52.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/DC34A8CB-46D3-DC11-981D-001617C3B6E2.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/F2338EC3-3FD3-DC11-A1A5-001617E30CA4.root')
    ),
    cosmics = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(2345678),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/140FBA0E-1599-DC11-B571-000423D99F3E.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/7A051E1C-4799-DC11-81FD-000423D99E46.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/9C659A18-4799-DC11-8899-000423D6C8E6.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/AADBC61C-4799-DC11-A162-000423D98AF0.root', 
            '/store/mc/2007/11/20/RelVal-RelValMinBias-1195584891/0000/B81B6DF7-2898-DC11-8ED5-001617E30F56.root')
    ),
    maxBunch = cms.int32(3),
    playback = cms.untracked.bool(False),
    minBunch = cms.int32(-5), ## in units of 25 nsec

    Label = cms.string(''),
    bunchspace = cms.int32(25), ## nsec

    beamhalo_plus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/28FE7F00-43D3-DC11-8217-000423D98EC4.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/B0D04F90-4CD3-DC11-9B60-001617E30D52.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/DC34A8CB-46D3-DC11-981D-001617C3B6E2.root', 
            '/store/relval/2008/2/4/RelVal-Minbias-1202115095/0000/F2338EC3-3FD3-DC11-A1A5-001617E30CA4.root')
    ),
    forwardDetectors = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('file:/data/uberthon/mm/767D5826-186A-DC11-88E3-001A928116E8.root')
    ),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(2.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('file:/data/uberthon/mm/767D5826-186A-DC11-88E3-001A928116E8.root')
    )
)


