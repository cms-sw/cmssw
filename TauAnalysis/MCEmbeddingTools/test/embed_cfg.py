# Auto generated configuration file
# using: 
# Revision: 1.381.2.2 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO --no_exec --conditions=FrontierConditions_GlobalTag,START53_V7A::All --fileout=embedded.root --python_filename=embed.py --customise=TauAnalysis/MCEmbeddingTools/embeddingCustomizeAll -n 10
import FWCore.ParameterSet.Config as cms

process = cms.Process('EmbeddedRECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Define input source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
        '/store/user/veelken/CMSSW_5_3_x/skims/simZmumu_madgraph_RECO_1_1_lTW.root'
    ),
##     eventsToProcess = cms.untracked.VEventRange(
##         '1:8516:3403697',
##         '1:8516:3403810',
##         '1:9906:3959466',
##         '1:9906:3959522',
##         '1:9906:3959549',
##         '1:13501:5396149',
##         '1:9174:3666598',
##         '1:9869:3944477',
##         '1:9869:3944479',
##         '1:9916:3963253',
##         '1:9916:3963401',
##         '1:9926:3967526',
##         '1:13686:5470217',
##         '1:16358:6537628',
##         '1:16358:6537632',
##         '1:16358:6537824',
##         '1:16358:6537833',
##         '1:58812:23505666',
##         '1:58812:23505821'
##     )
)

process.options = cms.untracked.PSet()

# Add Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.22 $'),
    annotation = cms.untracked.string('TauAnalysis/MCEmbeddingTools/python/PFEmbeddingSource_cff nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Define output files
process.outputFiles = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    ##fileName = cms.untracked.string('embed_AOD.root'),
    fileName = cms.untracked.string('/data1/veelken/CMSSW_5_3_x/skims/simDYmumu_embedded_mutau_2013Mar26_wNoise_AOD.root'),
    ##fileName = cms.untracked.string('/data1/veelken/CMSSW_5_3_x/skims/simDYmumu_embedded_mutau_2013Mar19_wNoise_woCaloRecHitMixing_AOD.root'),                                   
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Customise HLT menu for running on MC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
process = customizeHLTforMC(process)

process.GlobalTag.globaltag = 'START53_V7A::All'

process.filterEmptyEv = cms.EDFilter("EmptyEventsFilter",
    src = cms.untracked.InputTag("generator","","HLT2"),
    target = cms.untracked.int32(1)
)

process.cleanedGeneralTracks = cms.EDProducer("MuonTrackCleaner",
    selectedMuons = cms.InputTag(""), # CV: replaced in embeddingCustomizeAll.py
    tracks = cms.VInputTag("generalTracks"),
    dRmatch = cms.double(3.e-1),
    removeDuplicates = cms.bool(True),
    type = cms.string("inner tracks"),
    verbosity = cms.int32(0)                                           
)
process.cleanedParticleFlow = cms.EDProducer("MuonPFCandidateCleaner",
    selectedMuons = cms.InputTag(""), # CV: replaced in embeddingCustomizeAll.py
    pfCands = cms.InputTag("particleFlow"),
    dRmatch = cms.double(3.e-1),
    removeDuplicates = cms.bool(True),                          
    verbosity = cms.int32(0)                                           
)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.generator = cms.EDProducer("MCParticleReplacer",
    src = cms.InputTag(""),
    beamSpotSrc = cms.InputTag("dummy"),
    hepMcMode = cms.string('new'),
    verbose = cms.bool(False),
    algorithm = cms.string('Ztautau'),
    hepMcSrc = cms.InputTag("generator"),
    primaryVertexLabel = cms.InputTag("dummy"),
    Ztautau = cms.PSet(
        TauolaOptions = cms.PSet(
            UseTauolaPolarization = cms.bool(False),
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak2 = cms.int32(0),
                pjak1 = cms.int32(0)
            )
        ),
        PhotosOptions = cms.PSet(),                             
        PythiaParameters = cms.PSet(
            pythiaUESettings = cms.vstring(
                'MSTJ(11)=3     ! Choice of the fragmentation function', 
                'MSTJ(22)=2     ! Decay those unstable particles', 
                'PARJ(71)=10 .  ! for which ctau  10 mm', 
                'MSTP(2)=1      ! which order running alphaS', 
                'MSTP(33)=0     ! no K factors in hard cross sections', 
                'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
                'MSTP(52)=2     ! work with LHAPDF', 
                'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
                'MSTP(82)=4     ! Defines the multi-parton model', 
                'MSTU(21)=1     ! Check on possible errors during program execution', 
                'PARP(82)=1.8387   ! pt cutoff for multiparton interactions', 
                'PARP(89)=1960. ! sqrts for which PARP82 is set', 
                'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
                'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
                'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
                'PARP(67)=2.5    ! amount of initial-state radiation', 
                'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
                'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
                'PARP(62)=1.25   ! ', 
                'PARP(64)=0.2    ! ', 
                'MSTP(91)=1      !', 
                'PARP(91)=2.1   ! kt distribution', 
                'PARP(93)=15.0  ! '
            ),
            parameterSets = cms.vstring('pythiaUESettings')                               
        ),                                   
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        beamEnergy = cms.double(4000.0),
	rfRotationAngle = cms.double(90.),   				   
        generatorMode = cms.string('Tauola'),
        enablePhotosFSR = cms.bool(False),       
        filterEfficiency = cms.untracked.double(1.0),
        minVisibleTransverseMomentum = cms.untracked.string(''),
        applyMuonRadiationCorrection = cms.string("photos"),                                       
	verbosity = cms.int32(0)     				   
    ),
    pluginType = cms.string('ParticleReplacerZtautau')
)

process.ProductionFilterSequence = cms.Sequence(process.cleanedGeneralTracks+process.cleanedParticleFlow+process.generator+process.filterEmptyEv)

# Path and EndPath definitions of RECO sequence
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.outputFiles_step = cms.EndPath(process.outputFiles)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.outputFiles_step])
# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

#--------------------------------------------------------------------------------
# Define customisation options

process.customization_options = cms.PSet(
    parseCommandLine             = cms.bool(False),    # enable reading of configuration parameter values by parsing command-line
    isMC                         = cms.bool(True),     # set to true for MC/false for data
    ZmumuCollection              = cms.InputTag('genMuonsFromZs'), # collection of selected Z->mumu candidates
    # CV: use inputProcessRECO = "RECO" and inputProcessSIM = "SIM" for samples from official production;
    #         inputProcessRECO = inputProcessSIM = "HLT" for test samples produced privately, using cmsDriver
    inputProcessRECO             = cms.string("RECO"), # instanceLabel to be used for retrieving collections of reconstructed objects reconstructed in original Z->mumu event
    inputProcessSIM              = cms.string("SIM"),  # instanceLabel to be used for retrieving collections of generator level objects in original Z->mumu event 
    cleaningMode                 = cms.string("DEDX"), # option for muon calo. cleaning: 'DEDX'=muon energy loss expected on average, 'PF'=actual energy deposits associated to PFMuon
    muonCaloCleaningSF           = cms.double(1.0),    # option for subtracting too much (muonCaloSF > 1.0) or too few (muonCaloSF < 1.0) calorimeter energy around muon,
                                                       # too be used for studies of systematic uncertainties
    muonTrackCleaningMode        = cms.int32(2),       # option for muon track cleaning: 1=remove at most one track/charged PFCandidate matching muon,
                                                       # 2=remove all tracks/charged PFCandidates matched to muon in dR
    mdtau                        = cms.int32(115),     # mdtau value passed to TAUOLA: 0=no tau decay mode selection
    useTauolaPolarization        = cms.bool(False),    # disable tau polarization effects in TAUOLA, weight events by weights computed by TauSpinner instead
    transformationMode           = cms.int32(1),       # transformation mode: 0=mumu->mumu, 1=mumu->tautau
    rfRotationAngle              = cms.double(90.),    # rotation angle around Z-boson direction, used when replacing muons by simulated taus    
    embeddingMode                = cms.string("RH"),   # embedding mode: 'PF'=particle flow embedding, 'RH'=recHit embedding
    replaceGenOrRecMuonMomenta   = cms.string("gen"),  # take momenta of generated tau leptons from: 'rec'=reconstructed muons, 'gen'=generator level muons
    applyMuonRadiationCorrection = cms.string("photos"), # should I correct the momementa of replaced muons for muon -> muon + photon radiation ?
                                                       # (""=no correction, "pythia"/"photos"=correction is applied using PYTHIA/PHOTOS)
    minVisibleTransverseMomentum = cms.string("elec1_9had1_15"), # generator level cut on visible transverse momentum (typeN:pT,[...];[...])
    useJson                      = cms.bool(False),    # should I enable event selection by JSON file ?
    overrideBeamSpot             = cms.bool(False),    # should I override beamspot in globaltag ?
    applyZmumuSkim               = cms.bool(False),    # should I apply the Z->mumu event selection cuts ?
    applyMuonRadiationFilter     = cms.bool(False),    # should I apply the filter to reject events with muon -> muon + photon radiation ?
    disableCaloNoise             = cms.bool(False),    # should I disable the simulation of calorimeter noise when simulating the detector response for the embedded taus ?
    applyRochesterMuonCorr       = cms.bool(False),    # should I apply muon momentum corrections determined by the Rochester group (documented in AN-12/298) ?
    skipCaloRecHitMixing         = cms.bool(False),    # disable mixing of calorimeter recHit collections
                                                       # WARNING: needs to be set to false for production samples !!
    muonMixingMode               = cms.int32(1)        # option for mixing hits and tracks in muon detectors: 1=mix recHits, run muon track segment and track reconstruction on mixed recHit collection;
                                                       # mix recHits, but mix tracks instead of rerunning track reconstruction on mixed recHit collection; 3=mix tracks, do not mix recHits
                                                       # WARNING: options 2 and 3 not thoroughly tested yet !!
)

# Define "hooks" for replacing configuration parameters
# in case running jobs on the CERN batch system/grid
#__process.customization_options.isMC = cms.bool($isMC)
#__process.customization_options.ZmumuCollection = cms.InputTag('$ZmumuCollection')
#__process.customization_options.mdtau = cms.int32($mdtau)
#__process.customization_options.minVisibleTransverseMomentum = cms.string("$minVisibleTransverseMomentum")
#__process.customization_options.rfRotationAngle = cms.double($rfRotationAngle)
#__process.customization_options.embeddingMode = cms.string("$embeddingMode")
#__process.customization_options.replaceGenOrRecMuonMomenta = cms.string("$replaceGenOrRecMuonMomenta")
#__process.customization_options.applyMuonRadiationCorrection = cms.string("$applyMuonRadiationCorrection")
#__process.customization_options.cleaningMode = cms.string("$cleaningMode")
#__process.customization_options.muonCaloCleaningSF = cms.double($muonCaloCleaningSF)
#__process.customization_options.muonTrackCleaningMode = cms.int32($muonTrackCleaningMode)
#__process.customization_options.applyZmumuSkim = cms.bool($applyZmumuSkim)
#__process.customization_options.applyMuonRadiationFilter = cms.bool($applyMuonRadiationFilter)
#__process.customization_options.disableCaloNoise = cms.bool($disableCaloNoise)
#__process.customization_options.applyRochesterMuonCorr = cms.bool($applyRochesterMuonCorr)
#__process.customization_options.overrideBeamSpot = cms.bool(not $isMC)
#
# CV: set name of output file to value defined in crab config (without any file paths)
#__process.outputFiles.fileName = cms.untracked.string('embed_AOD.root')
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Apply customisation options
from TauAnalysis.MCEmbeddingTools.embeddingCustomizeAll import customise 
process = customise(process)
#--------------------------------------------------------------------------------

# after processing RECO sequence for 1st event, print event content
##process.printEventContent = cms.EDAnalyzer("EventContentAnalyzer")
##process.filterFirstEvent = cms.EDFilter("EventCountFilter",
##    numEvents = cms.int32(1)
##)
##process.printFirstEventContentPath = cms.Path(process.filterFirstEvent + process.printEventContent)

##process.schedule.extend([process.printFirstEventContentPath])

##process.outputFiles.outputCommands = cms.untracked.vstring('keep *')

##process.dumpMuons = cms.EDAnalyzer("DumpMuons",
##    src = cms.InputTag('muons'),
##    minPt = cms.double(8.),
##)
##process.dumpMuonsPath = cms.Path(process.dumpMuons)
##
##process.schedule.extend([process.dumpMuonsPath])

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

processDumpFile = open('embed.dump', 'w')
print >> processDumpFile, process.dumpPython()
