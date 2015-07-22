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
# we have put minbias files for all the sources, just as an example
#
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in units of 25 nsec

    bunchspace = cms.int32(25), ## nsec
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(10.)
        ),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/18890F4C-FD99-DD11-BFF9-000423D996C8.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/1A423C16-6099-DD11-9320-000423D9853C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/5EDA8A7F-5D99-DD11-B1CD-001617C3B706.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/600D1E6A-5F99-DD11-A7D5-000423D9890C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/68ECAE92-5F99-DD11-ACAB-000423D98E6C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/8802D325-5E99-DD11-B858-000423D98A44.root')
    ),
    cosmics = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.6625e-05)
        ),
        seed = cms.int32(2345678),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/18890F4C-FD99-DD11-BFF9-000423D996C8.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/1A423C16-6099-DD11-9320-000423D9853C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/5EDA8A7F-5D99-DD11-B1CD-001617C3B706.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/600D1E6A-5F99-DD11-A7D5-000423D9890C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/68ECAE92-5F99-DD11-ACAB-000423D98E6C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/8802D325-5E99-DD11-B858-000423D98A44.root')
     ),
    beamhalo_minus = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(0.00040503)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/18890F4C-FD99-DD11-BFF9-000423D996C8.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/1A423C16-6099-DD11-9320-000423D9853C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/5EDA8A7F-5D99-DD11-B1CD-001617C3B706.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/600D1E6A-5F99-DD11-A7D5-000423D9890C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/68ECAE92-5F99-DD11-ACAB-000423D98E6C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/8802D325-5E99-DD11-B858-000423D98A44.root')
     ),
    beamhalo_plus = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(0.00040503)
        ),
        seed = cms.int32(3456789),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/18890F4C-FD99-DD11-BFF9-000423D996C8.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/1A423C16-6099-DD11-9320-000423D9853C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/5EDA8A7F-5D99-DD11-B1CD-001617C3B706.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/600D1E6A-5F99-DD11-A7D5-000423D9890C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/68ECAE92-5F99-DD11-ACAB-000423D98E6C.root',
        '/store/relval/CMSSW_2_1_10/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/8802D325-5E99-DD11-B858-000423D98A44.root')

    ),
    mixObjects = cms.PSet(theMixObjects)
)


