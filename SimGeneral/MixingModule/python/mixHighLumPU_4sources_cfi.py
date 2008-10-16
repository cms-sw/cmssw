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
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in units of 25 nsec

    bunchspace = cms.int32(25), ## nsec

    playback = cms.untracked.bool(False),
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(10.)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/16D19E34-1650-DD11-99A0-000423D99A8E.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/28A8E53E-2750-DD11-B3C5-001617C3B5D8.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/A4D46D32-2B50-DD11-8806-001D09F2910A.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/BED002A6-2A50-DD11-9F65-001D09F252F3.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/C81A2549-2A50-DD11-8176-001D09F24EC0.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/F6808575-1350-DD11-AB99-000423D6B48C.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/FE15082B-1B50-DD11-A8EB-000423D987FC.root')
    ),
    cosmics = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.6625e-05)
        ),
        seed = cms.int32(2345678),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/16D19E34-1650-DD11-99A0-000423D99A8E.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/28A8E53E-2750-DD11-B3C5-001617C3B5D8.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/A4D46D32-2B50-DD11-8806-001D09F2910A.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/BED002A6-2A50-DD11-9F65-001D09F252F3.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/C81A2549-2A50-DD11-8176-001D09F24EC0.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/F6808575-1350-DD11-AB99-000423D6B48C.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/FE15082B-1B50-DD11-A8EB-000423D987FC.root')
    ),
    beamhalo_minus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(0.00040503)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/16D19E34-1650-DD11-99A0-000423D99A8E.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/28A8E53E-2750-DD11-B3C5-001617C3B5D8.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/A4D46D32-2B50-DD11-8806-001D09F2910A.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/BED002A6-2A50-DD11-9F65-001D09F252F3.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/C81A2549-2A50-DD11-8176-001D09F24EC0.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/F6808575-1350-DD11-AB99-000423D6B48C.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/FE15082B-1B50-DD11-A8EB-000423D987FC.root')
    ),
    beamhalo_plus = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(0.00040503)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/16D19E34-1650-DD11-99A0-000423D99A8E.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/28A8E53E-2750-DD11-B3C5-001617C3B5D8.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/A4D46D32-2B50-DD11-8806-001D09F2910A.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/BED002A6-2A50-DD11-9F65-001D09F252F3.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/C81A2549-2A50-DD11-8176-001D09F24EC0.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/F6808575-1350-DD11-AB99-000423D6B48C.root',
                                          '/store/relval/2008/7/11/RelVal-RelValMinBias-1215820540/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/CMSSW_2_1_0_pre8-RelVal-1215820540-STARTUP_V4-unmerged/0000/FE15082B-1B50-DD11-A8EB-000423D987FC.root')
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


