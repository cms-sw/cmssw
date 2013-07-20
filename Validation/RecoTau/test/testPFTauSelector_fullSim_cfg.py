import FWCore.ParameterSet.Config as cms


process = cms.Process("TAU2")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'rfio:/castor/cern.ch/user/p/pjanot/CMSSW219/reco_QCDpt20_30_Full.root',
#'rfio:/castor/cern.ch/user/p/pjanot/CMSSW219/reco_QCDpt30_50_Full.root',
#'rfio:/castor/cern.ch/user/p/pjanot/CMSSW219/reco_QCDpt50_80_Full.root',
#'rfio:/castor/cern.ch/user/p/pjanot/CMSSW219/reco_QCDpt80_120_Full.root'
    )
)

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")


# Conditions: fake or frontier
# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'

process.DQMStore = cms.Service("DQMStore")

process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

process.TauMCProducer  = cms.EDProducer("HLTTauMCProducer",
                              GenParticles  = cms.untracked.InputTag("source"),
                              ptMinTau      = cms.untracked.double(5),
                              ptMinMuon     = cms.untracked.double(14),
                              ptMinElectron = cms.untracked.double(12),
                              BosonID       = cms.untracked.vint32(23),
                              EtaMax         = cms.untracked.double(2.5)
)

#process.fixedConePFTauProducer.LeadChargedHadrCand_minPt = cms.double(1.0)
#process.shrinkingConePFTauProducer.JetPtMin = cms.double(15.0)

process.shrinkingConePFTauDiscriminationByChargeIsolation = cms.EDProducer("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    PFTauProducer = cms.InputTag('shrinkingConePFTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    # following parameters are considered when ManipulateTracks_insteadofChargedHadrCands paremeter is set true
    # *BEGIN*
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    TrackerIsolAnnulus_Candsmaxn = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn = cms.int32(0)
)

process.shrinkingConePFTauDiscriminationByGammaIsolation = cms.EDProducer("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    PFTauProducer = cms.InputTag('shrinkingConePFTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    # following parameters are considered when ManipulateTracks_insteadofChargedHadrCands paremeter is set true
    # *BEGIN*
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    TrackerIsolAnnulus_Candsmaxn = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn = cms.int32(0)
)

process.shrinkingConePFTauDiscriminationByNeutralHadrons = cms.EDProducer("PFRecoTauDiscriminationByNeutralHadrons",
                                                                               PFTauProducer = cms.InputTag('shrinkingConePFTauProducer'),
                                                                               NumberOfAllowedNeutralHadronsInSignalCone = cms.int32(0)                                                                     
)

process.load("RecoJets.Configuration.RecoPFJets_cff")
process.load("Validation/RecoTau/TauTagValidation_cfi")
process.load("Validation/RecoTau/TauTagValidationGenJets_cfi")
process.load("Validation.RecoTau.GenJetRefProducer_cfi")

process.generatorLevelJets.TauProducer = cms.string('shrinkingConePFTauProducer')
process.generatorLevelJets.discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByNeutralHadrons"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByChargeIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByGammaIsolation"),selectionCut = cms.double(0.5))
 )


#process.p1 = cms.Path(
    *process.PFTau
#    *process.shrinkingConePFTauDiscriminationByChargeIsolation
#    *process.shrinkingConePFTauDiscriminationByGammaIsolation
#    *process.shrinkingConePFTauDiscriminationByNeutralHadrons
#    *process.GenJetProducer
#    *process.generatorLevelJets
    
#    *process.TauMCProducer
#    *process.TauRefCombiner
 #   )


process.load("Configuration.EventContent.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
#    process.AODSIMEventContent,
    fileName = cms.untracked.string('/build1/gennai/aodFullSimQCDJets_20-30.root'),
                                outputCommands = cms.untracked.vstring(
#    "drop *",
    "keep *"
#    "keep *_generalTracks_*_*",
#    "keep recoPFJets_*_*_*",
#    "keep recoPFCandidates_*_*_*"
    )

)

process.outpath = cms.EndPath(process.aod)

#
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
