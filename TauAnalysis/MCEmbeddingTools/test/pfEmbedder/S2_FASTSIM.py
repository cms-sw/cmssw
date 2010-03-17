# Auto generated configuration file
# using: 
# Revision: 1.151 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: -s FASTSIM --no_exec --filein=zMuMuEmbed_output.root --conditions=STARTUP3X_V8A::All
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT2')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('FastSimulation/Configuration/RandomServiceInitialization_cff')
process.load('FastSimulation.PileUpProducer.PileUpSimulator10TeV_cfi')
process.load('FastSimulation/Configuration/FamosSequences_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('FastSimulation/Configuration/FamosSequences_cff')
process.load('FastSimulation/Configuration/HLT_8E29_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
#process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('FastSimulation/Configuration/CommonInputs_cff')
process.load('FastSimulation/Configuration/EventContent_cff')



process.famosSimulationSequence.remove(process.offlineBeamSpot)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.151 $'),
    annotation = cms.untracked.string('-s nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-10)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/fruboes/Zmumu/zMuMuEmbed_output.root')
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.RECOSIMEventContent.outputComma ds,
    fileName = cms.untracked.string('file:/tmp/fruboes/Zmumu/s2_FASTSIM.root'),
#     fileName = cms.untracked.string('s2_FASTSIM.root'),
#fileName = cms.untracked.string('file:/tmp/fruboes/Zmumu/Bs2_FASTSIM.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.famosPileUp.PileUpSimulator = process.PileUpSimulatorBlock.PileUpSimulator
process.famosPileUp.PileUpSimulator.averageNumber = 0
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosSimHits.ActivateDecays.comEnergy = 10000
process.simulation = cms.Sequence(process.simulationWithFamos)
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# set correct vertex smearing
# XXX
#process.Early10TeVCollisionVtxSmearingParameters.type = cms.string("BetaFunc")
#process.famosSimHits.VertexGenerator = process.Early10TeVCollisionVtxSmearingParameters
#process.famosPileUp.VertexGenerator = process.Early10TeVCollisionVtxSmearingParameters
  
process.FlatVtxSmearingParameters.type = cms.string("Flat")
process.famosSimHits.VertexGenerator = process.FlatVtxSmearingParameters
process.famosPileUp.VertexGenerator = process.FlatVtxSmearingParameters
  
# Apply ECAL/HCAL miscalibration
process.ecalRecHit.doMiscalib = True
process.hbhereco.doMiscalib = True
process.horeco.doMiscalib = True
process.hfreco.doMiscalib = True
# Apply Tracker and Muon misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True

process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True

#process.GlobalTag.globaltag = 'STARTUP3X_V8A::All'
process.GlobalTag.globaltag = 'STARTUP31X_V4::All' # 31x
#process.GlobalTag.globaltag = 'MC_31X_V5::All' # 31x

# Path and EndPath definitions
process.reconstruction = cms.Path(process.reconstructionWithFamos)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.reconstruction,process.out_step])
