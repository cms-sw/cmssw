# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM --geometry GEMCosmicStand --eventcontent FEVTDEBUGHLT --era phase2_muon -n 100 --fileout out_reco.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.phase2_muon)

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
process.load('Validation.MuonGEMHits.MuonGEMHits_cff')
process.load('Validation.MuonGEMDigis.MuonGEMDigis_cff')
process.load('Validation.MuonGEMRecHits.MuonGEMRecHits_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

#process.load('Configuration.StandardSequences.Validation_cff')
#process.load('Configuration.StandardSequences.Harvesting_cff')
#process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
#process.load('DQMServices.Components.EDMtoMEConverter_cff')

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
    fileName = cms.untracked.string('out_reco.root'),
    outputCommands = cms.untracked.vstring( ('keep *')),
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:dqm.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

#process.FEVTDEBUGHLToutput.outputCommands.append()

# Additional output definition
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Cosmic Muon generator
process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENproducer_cfi")
process.generator.TrackerOnly = True
process.generator.MinP = 1000
process.generator.RadiusOfTarget = cms.double(10.0)#in cm
process.generator.ZDistOfTarget = cms.double(10.0) #in cm

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
                                       trackChi2 = cms.double(100.0),
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

#process.gemSimHitValidation.detailPlot = cms.bool(True)
#process.gemSimTrackValidation.detailPlot = cms.bool(True)
#process.gemStripValidation.detailPlot = cms.bool(True)
#process.gemDigiTrackValidation.detailPlot = cms.bool(True)
#process.gemRecHitsValidation.detailPlot = cms.bool(True)
#process.gemRecHitTrackValidation.detailPlot = cms.bool(True)

process.rawDataCollector.RawCollectionList = cms.VInputTag(cms.InputTag("gemPacker"))
# Path and EndPath definitions
process.generation_step = cms.Path(process.generator+process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digi2raw_step = cms.Path(process.gemPacker+process.rawDataCollector)
process.raw2digi_step = cms.Path(process.muonGEMDigis)
process.digitisation_step = cms.Path(process.randomEngineStateProducer+process.mix+process.simMuonGEMDigis)
process.reconstruction_step = cms.Path(process.gemLocalReco+process.GEMCosmicMuon+process.GEMCosmicMuonInSide)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
#process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.validation_step = cms.Path(process.gemSimValidation
                                +process.gemStripValidation+process.gemDigiTrackValidation
                                +process.gemLocalRecoValidation)
process.dqmoffline_step = cms.EndPath(process.DQMOffline)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,
                                process.digitisation_step,
                                process.digi2raw_step,process.raw2digi_step,process.reconstruction_step,
                                process.validation_step,#process.endjob_step,
                                #process.genHarvesting,
                                process.DQMoutput_step,
                                process.FEVTDEBUGHLToutput_step
                                )


process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
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
