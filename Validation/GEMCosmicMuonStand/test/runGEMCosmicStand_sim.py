# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM --geometry GEMCosmicStand --eventcontent FEVTDEBUGHLT --era phase2_muon -n 100 --fileout out_reco.root
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('DIGI',eras.phase2_muon)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Geometry.GEMGeometry.GeometryGEMCosmicStand_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('SimMuon.GEMDigitizer.muonGEMDigi_cff')
process.load('EventFilter.GEMRawToDigi.gemPacker_cfi')
process.load('EventFilter.RawDataCollector.rawDataCollector_cfi')
process.load('EventFilter.GEMRawToDigi.muonGEMDigis_cfi')
process.load('RecoLocalMuon.GEMRecHit.gemLocalReco_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5000))

# Input source
process.source = cms.Source("EmptySource")
process.options = cms.untracked.PSet()

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('CosmicMuonGenerator nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('out_digi.root'),
    outputCommands = cms.untracked.vstring( ('keep *')),
    splitLevel = cms.untracked.int32(0)
)

#process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
#    dataset = cms.untracked.PSet(
#        dataTier = cms.untracked.string('DQMIO'),
#        filterName = cms.untracked.string('')
#    ),
#    fileName = cms.untracked.string('file:dqm.root'),
#    outputCommands = process.DQMEventContent.outputCommands,
#    splitLevel = cms.untracked.int32(0)
#)

#process.FEVTDEBUGHLToutput.outputCommands.append()

# Additional output definition
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Cosmic Muon generator
#process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENproducer_cfi")
#process.generator.RadiusOfTarget = cms.double(120.0)#in mm
#process.generator.ZDistOfTarget = cms.double(280.0)
#process.generator.MinP = 100
#process.generator.ElossScaleFactor = 0.0
#process.generator.TrackerOnly = True
#process.generator.PlugVz = cms.double(0.0) #[mm]
#process.generator.RhoAir = cms.double(0.0)
#process.generator.RhoWall = cms.double(0.0)
#process.generator.RhoRock = cms.double(0.0)
#process.generator.RhoClay = cms.double(0.0)
#process.generator.RhoPlug = cms.double(0.0)
#process.generator.ClayWidth = cms.double(0.0)
#process.generator.Verbosity = cms.bool(True)

process.generator = cms.EDProducer("CosmicGun",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        MaxEta = cms.double(0.1),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(100.01),
        MinEta = cms.double(-0.1),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(3),
        PartID = cms.vint32(-13)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single mu pt 100')
)

process.RandomNumberGeneratorService.generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )

process.mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    bunchspace = cms.int32(450),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    digitizers = cms.PSet(),
    
    mixObjects = cms.PSet(
        mixSH = cms.PSet(
            crossingFrames = cms.untracked.vstring('MuonGEMHits'),
            input = cms.VInputTag(cms.InputTag("g4SimHits","MuonGEMHits")),
            type = cms.string('PSimHit'),
            subdets = cms.vstring('MuonGEMHits'),            
            )
        ),
    mixTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('SimTrack')
    ),
)

process.g4SimHits.UseMagneticField = cms.bool(False)
process.load('RecoMuon.TrackingTools.MuonServiceProxy_cff')
process.MuonServiceProxy.ServiceParameters.Propagators.append('StraightLinePropagator')

process.GEMCosmicMuon = cms.EDProducer("GEMCosmicMuon",
                                       process.MuonServiceProxy,
                                       gemRecHitLabel = cms.InputTag("gemRecHits"),
                                       doInnerSeeding = cms.bool(False),
                                       trackChi2 = cms.double(10000.0),
                                       trackResX = cms.double(5.0),
                                       trackResY = cms.double(15.0),
                                       MuonSmootherParameters = cms.PSet(
                                           PropagatorAlong = cms.string('SteppingHelixPropagatorAny'),
                                           PropagatorOpposite = cms.string('SteppingHelixPropagatorAny'),
                                           RescalingFactor = cms.double(5.0)
                                           ),
                                       )
process.GEMCosmicMuon.ServiceParameters.GEMLayers = cms.untracked.bool(True)
process.GEMCosmicMuon.ServiceParameters.CSCLayers = cms.untracked.bool(False)
process.GEMCosmicMuon.ServiceParameters.RPCLayers = cms.bool(False)
process.GEMCosmicMuonInSide = process.GEMCosmicMuon.clone(doInnerSeeding = cms.bool(True))

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)
process.load('Validation.GEMCosmicMuonStand.GEMCosmicMuonStandEfficiency_cff')
process.load('Validation.GEMCosmicMuonStand.GEMCosmicMuonStandSim_cff')

process.rawDataCollector.RawCollectionList = cms.VInputTag(cms.InputTag("gemPacker"))
# Path and EndPath definitions
process.genParticles.src = cms.InputTag("generator:unsmeared")
process.g4SimHits.HepMCProductLabel = cms.InputTag("generator:unsmeared")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag("generator:unsmeared")
process.generation_step = cms.Path(process.generator+process.genParticles)
process.simulation_step = cms.Path(process.psim)
process.digi2raw_step = cms.Path(process.gemPacker+process.rawDataCollector)
process.raw2digi_step = cms.Path(process.muonGEMDigis)
process.digitisation_step = cms.Path(process.randomEngineStateProducer+process.mix+process.simMuonGEMDigis+process.gemLocalReco)
process.reconstruction_step = cms.Path(process.GEMCosmicMuon+process.GEMCosmicMuonInSide)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
#process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.validation_step = cms.Path(process.gemCosmicMuonStandSim+process.gemCosmicMuonStandEfficiency)
process.dqmoffline_step = cms.EndPath(process.DQMOffline)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)
# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,
                                process.digitisation_step,
                                process.digi2raw_step,process.raw2digi_step,
                                #process.reconstruction_step,
                                #process.validation_step,
                                #process.DQMoutput_step,
                                process.FEVTDEBUGHLToutput_step
                                )


process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
)

process.gemSegments.maxRecHitsInCluster = cms.int32(10)
process.gemSegments.minHitsPerSegment = cms.uint32(3)
process.gemSegments.clusterOnlySameBXRecHits = cms.bool(True)
process.gemSegments.dEtaChainBoxMax = cms.double(1.05)
process.gemSegments.dPhiChainBoxMax = cms.double(1.12)
process.gemSegments.dXclusBoxMax = cms.double(10.0)
process.gemSegments.dYclusBoxMax = cms.double(50.0)
process.gemSegments.preClustering = cms.bool(False)
process.gemSegments.preClusteringUseChaining = cms.bool(False)

process.simMuonGEMDigis.averageEfficiency = cms.double(0.98)
process.simMuonGEMDigis.averageNoiseRate = cms.double(0.0)
process.simMuonGEMDigis.doBkgNoise = cms.bool(False)
process.simMuonGEMDigis.doNoiseCLS = cms.bool(False)
process.simMuonGEMDigis.simulateElectronBkg = cms.bool(False)


#process.SteppingHelixPropagatorAny.debug = cms.bool(True)
#process.SteppingHelixPropagatorAny.sendLogWarning = cms.bool(True)
#process.SteppingHelixPropagatorAny.useInTeslaFromMagField = cms.bool(False)
#process.SteppingHelixPropagatorAny.useMagVolumes = cms.bool(False)
