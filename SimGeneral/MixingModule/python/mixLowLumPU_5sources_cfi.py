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
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDFilter("MixingModule",
    LabelPlayback = cms.string(''),
    beamhalo_minus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
    ),
    cosmics = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(2345678),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
    ),
    maxBunch = cms.int32(3),
    playback = cms.untracked.bool(False),
    minBunch = cms.int32(-5), ## in units of 25 nsec

    bunchspace = cms.int32(25), ## nsec

    beamhalo_plus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
    ),
    forwardDetectors = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(5.0)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
    ),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(2.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
    ),
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)


