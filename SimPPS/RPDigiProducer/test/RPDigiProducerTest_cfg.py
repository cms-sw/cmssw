import FWCore.ParameterSet.Config as cms

process = cms.Process("RPDigiProducerTest")

# Specify the maximum events to simulate
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Configure the output module (save the result in a file)
# Configure the output module
process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:RPDigiProducerTest_output.root')
)


process.load("SimGeneral.HepPDTESSource.pdt_cfi")


# Configure if you want to detail or simple log information.
# LoggerMax -- detail log info output including: errors.log, warnings.log, infos.log, debugs.log
# LoggerMin -- simple log info output to the standard output (e.g. screen)
process.load("Configuration.TotemCommon.LoggerMin_cfi")


################## STEP 1
process.source = cms.Source("EmptySource")

################## STEP 2 - process.generator

# Use random number generator service
process.load("Configuration.TotemCommon.RandomNumbers_cfi")

# Monte Carlo gun - elastic specific
energy = "7000"
import IOMC.Elegent.ElegentSource_cfi
process.generator = IOMC.Elegent.ElegentSource_cfi.generator
process.generator.fileName = IOMC.Elegent.ElegentSource_cfi.ElegentDefaultFileName(energy)

# particle generator paramteres
process.generator.t_min = '6E-2'  # beta* specific
process.generator.t_max = '6E-1'  # beta* specific
energy = "1180"


################## STEP 3 process.SmearingGenerator

# declare optics parameters
# process.load("Configuration.TotemOpticsConfiguration.OpticsConfig_7000GeV_90_cfi")

# Smearing
process.load("IOMC.SmearingGenerator.SmearingGenerator_cfi")

################## STEP 4 process.OptInfo

process.OptInfo = cms.EDAnalyzer("OpticsInformation")

process.load("Configuration.TotemOpticsConfiguration.OpticsConfig_1180GeV_11_cfi")

################## STEP 5 process.*process.g4SimHits

# Geometry - beta* specific
# process.load("Configuration.TotemCommon.geometryRP_cfi")
# process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/TotemRPData/data/RP_Beta_90_150_out/RP_Dist_Beam_Cent.xml')

# Magnetic Field, by default we have 3.8T
process.load("Configuration.StandardSequences.MagneticField_cff")

# G4 simulation & proton transport
process.load("Configuration.TotemCommon.g4SimHits_cfi")
#process.g4SimHits.Physics.BeamProtTransportSetup = process.BeamProtTransportSetup
process.g4SimHits.Generator.HepMCProductLabel = 'generator'    # The input source for G4 module is connected to "process.source".

process.load("Configuration.TotemCommon.geometryRP_cfi")
process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/TotemRPData/data/RP_1180_Beta_11_220/RP_Dist_Beam_Cent.xml')

process.g4SimHits.Physics.BeamProtTransportSetup = process.BeamProtTransportSetup



################## STEP 6 process.mix*process.RPSiDetDigitizer 

# No pile up for the mixing module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

########################### DIGI+RECO RP ##########################################

# process.load("SimPPS.RPDigiProducer.RPSiDetConf_cfi")



process.p1 = cms.Path(process.generator
                      *process.SmearingGenerator
                      *process.g4SimHits
#                       *process.mix
#                       *process.RPSiDetDigitizer
                      )

process.outpath = cms.EndPath(process.o1)

# print process.dumpConfig()
