import FWCore.ParameterSet.Config as cms

process = cms.Process("TestFlatGun")

# Specify the maximum events to simulate
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Configure the output module (save the result in a file)
process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *','drop *_*mix*_*_*', 'drop *_*_*Muon*_*', 'drop *_*_*Ecal*_*', 'drop *_*_*Hcal*_*', 'drop *_*_*Calo*_*', 'drop *_*_*Castor*_*', 'drop *_*_*FP420SI_*', 'drop *_*_*ZDCHITS_*', 'drop *_*_*BSCHits_*', 'drop *_*_*ChamberHits_*', 'drop *_*_*FibreHits_*', 'drop *_*_*WedgeHits_*','drop Sim*_*_*_*','drop edm*_*_*_*'),
    fileName = cms.untracked.string('file:MYtest44_.root')
)
process.outpath = cms.EndPath(process.o1)

# Configure if you want to detail or simple log information.
# LoggerMax -- detail log info output including: errors.log, warnings.log, infos.log, debugs.log
# LoggerMin -- simple log info output to the standard output (e.g. screen)
#process.load("Configuration.TotemCommon.LoggerMax_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('MYwarnings44',
        'MYerrors44',
 #       'MYinfos44',
        'MYdebugs44'),
    categories = cms.untracked.vstring('ForwardSim',
        'TotemRP'),
    debugModules = cms.untracked.vstring('*'),
    MYerrors44 = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    MYwarnings44 = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
 #   MYinfos44 = cms.untracked.PSet(
  #      threshold = cms.untracked.string('INFO')
  #  ),
    MYdebugs44 = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TotemRP = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        ),
        ForwardSim = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        )
    )
)

################## STEP 1 - process.generator
process.source = cms.Source("EmptySource")

# Use random number generator service
process.load("Configuration.TotemCommon.RandomNumbers_cfi")
process.RandomNumberGeneratorService.RPixDetDigitizer = cms.PSet(initialSeed =cms.untracked.uint32(137137))
# particle generator paramteres
process.load("IOMC.FlatProtonLogKsiLogTGun.Beta90Energy6500GeV_cfi")
process.generator.MinT = cms.untracked.double(-0.25)
process.generator.MaxT = cms.untracked.double(-0.55)
process.generator.MinPhi = cms.untracked.double(-2.099)
process.generator.MaxPhi = cms.untracked.double(-2.099)
process.generator.MinKsi = cms.untracked.double(-0.06)
process.generator.MaxKsi = cms.untracked.double(-0.065)

process.generator.RightArm = cms.untracked.bool(True)
process.generator.LeftArm = cms.untracked.bool(True)


process.generator.Verbosity = cms.untracked.int32(0)
#process.generator.MinT = cms.untracked.double(-0.0001)
#process.generator.MaxT = cms.untracked.double(-20.0)
#process.generator.MinPhi = cms.untracked.double(-3.141592654)
#process.generator.MaxPhi = cms.untracked.double(3.141592654)
#process.generator.MinKsi = cms.untracked.double(-0.0001)
#process.generator.MaxKsi = cms.untracked.double(-0.3)

################## STEP 2 process.SmearingGenerator

# declare optics parameters
process.load("Configuration.TotemOpticsConfiguration.OpticsConfig_6500GeV_0p8_145urad_cfi")

# Smearing
process.load("IOMC.SmearingGenerator.SmearingGenerator_cfi")

################## STEP 3 process.g4SimHits

# Geometry - beta* specific
process.load("Configuration.TotemCommon.geometryRP_CTPPS_cfi")

# TODO Change to the LowBetaSettings
process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/TotemRPData/data/RP_Beta_90/RP_Dist_Beam_Cent.xml')

# misalignments
process.load("TotemAlignment.RPDataFormats.TotemRPIncludeAlignments_cfi")
process.TotemRPIncludeAlignments.MisalignedFiles = cms.vstring()

# Magnetic Field, by default we have 3.8T
process.load("Configuration.StandardSequences.MagneticField_cff")

# G4 simulation & proton transport
process.load("Configuration.TotemCommon.g4SimHits_cfi")
process.g4SimHits.Physics.BeamProtTransportSetup = process.BeamProtTransportSetup
process.g4SimHits.Generator.HepMCProductLabel = 'generator'    # The input source for G4 module is connected to "process.source".
process.g4SimHits.G4TrackingManagerVerbosity = cms.untracked.int32(0)

# Use particle table
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.g4SimHits.CTPPSSD = cms.PSet(
 Verbosity = cms.untracked.int32(0)
)

################## Step 3 - Magnetic field configuration
# todo declare in standard way (not as hardcoded raw config)

process.magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms.xml',
        'Geometry/CMSCommonData/data/cmsMagneticField.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)
process.prefer("magfield")

process.ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_8T')
    ),
    label = cms.untracked.string('parametrizedField')
)

process.VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    scalingVolumes = cms.vint32(14100, 14200, 17600, 17800, 17900,
        18100, 18300, 18400, 18600, 23100,
        23300, 23400, 23600, 23800, 23900,
        24100, 28600, 28800, 28900, 29100,
        29300, 29400, 29600, 28609, 28809,
        28909, 29109, 29309, 29409, 29609,
        28610, 28810, 28910, 29110, 29310,
        29410, 29610, 28611, 28811, 28911,
        29111, 29311, 29411, 29611),
    scalingFactors = cms.vdouble(1, 1, 0.994, 1.004, 1.004,
        1.005, 1.004, 1.004, 0.994, 0.965,
        0.958, 0.958, 0.953, 0.958, 0.958,
        0.965, 0.918, 0.924, 0.924, 0.906,
        0.924, 0.924, 0.918, 0.991, 0.998,
        0.998, 0.978, 0.998, 0.998, 0.991,
        0.991, 0.998, 0.998, 0.978, 0.998,
        0.998, 0.991, 0.991, 0.998, 0.998,
        0.978, 0.998, 0.998, 0.991),
    useParametrizedTrackerField = cms.bool(True),
    label = cms.untracked.string(''),
    version = cms.string('grid_1103l_090322_3_8t'),
    debugBuilder = cms.untracked.bool(False),
    paramLabel = cms.string('parametrizedField'),
    geometryVersion = cms.int32(90322),
    gridFiles = cms.VPSet(cms.PSet(
        path = cms.string('grid.[v].bin'),
        master = cms.int32(1),
        volumes = cms.string('1-312'),
        sectors = cms.string('0')
    ),
        cms.PSet(
            path = cms.string('S3/grid.[v].bin'),
            master = cms.int32(3),
            volumes = cms.string('176-186,231-241,286-296'),
            sectors = cms.string('3')
        ),
        cms.PSet(
            path = cms.string('S4/grid.[v].bin'),
            master = cms.int32(4),
            volumes = cms.string('176-186,231-241,286-296'),
            sectors = cms.string('4')
        ),
        cms.PSet(
            path = cms.string('S9/grid.[v].bin'),
            master = cms.int32(9),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('9')
        ),
        cms.PSet(
            path = cms.string('S10/grid.[v].bin'),
            master = cms.int32(10),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('10')
        ),
        cms.PSet(
            path = cms.string('S11/grid.[v].bin'),
            master = cms.int32(11),
            volumes = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors = cms.string('11')
        )),
    cacheLastVolume = cms.untracked.bool(True)
)

################## STEP 4 mix pdt_cfi

process.load("Configuration.TotemCommon.mixNoPU_cfi")

# Use particle table
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

################## STEP 5 RPDigiProducer & RPix

process.load("SimTotem.RPDigiProducer.RPSiDetConf_cfi")
process.load("SimPPS.CTPPSPixelDigiProducer.RPixDetConf_cfi")

################## STEP 6 reco

process.load("Configuration.TotemStandardSequences.RP_Digi_and_TrackReconstruction_cfi")

################## STEP 7 TotemNtuplizer

process.load("TotemAnalysis.TotemNtuplizer.TotemNtuplizer_cfi")
process.TotemNtuplizer.outputFileName = "test.ntuple.root"
process.TotemNtuplizer.RawEventLabel = 'source'
process.TotemNtuplizer.RPReconstructedProtonCollectionLabel = cms.InputTag('RP220Reconst')
process.TotemNtuplizer.RPReconstructedProtonPairCollectionLabel = cms.InputTag('RP220Reconst')
process.TotemNtuplizer.RPMulFittedTrackCollectionLabel = cms.InputTag("RPMulTrackNonParallelCandCollFit")
process.TotemNtuplizer.includeDigi = cms.bool(True)
process.TotemNtuplizer.includePatterns = cms.bool(True)

process.digiAnal = cms.EDAnalyzer("CTPPSPixelDigiAnalyzer",
      label=cms.untracked.string("RPixDetDigitizer"),
     Verbosity = cms.int32(0),
   RPixVerbosity = cms.int32(0),
   RPixActiveEdgeSmearing = cms.double(0.020),
    RPixActiveEdgePosition = cms.double(0.150)
)
########

process.p1 = cms.Path(
	process.generator
#	*process.SmearingGenerator
	*process.g4SimHits
	*process.mix
#	*process.RPSiDetDigitizer 
*process.RPixDetDigitizer
#*process.digiAnal
#	*process.RPClustProd
#	*process.RPHecoHitProd
#	*process.RPSinglTrackCandFind
#	*process.RPSingleTrackCandCollFit


#	*process.RP220Reconst
#	*process.TotemNtuplizer

)


