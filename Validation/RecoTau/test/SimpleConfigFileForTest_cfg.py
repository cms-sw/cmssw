import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_3_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP31X_V4-v1/0012/FED16463-E295-DE11-9BBE-000423D94494.root', 
        '/store/relval/CMSSW_3_3_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP31X_V4-v1/0012/CE1E7DD0-0896-DE11-8133-0019B9F705A3.root', 
        '/store/relval/CMSSW_3_3_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP31X_V4-v1/0012/BC329B63-DD95-DE11-9656-003048D37456.root', 
        '/store/relval/CMSSW_3_3_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP31X_V4-v1/0012/7C9552D0-DC95-DE11-AE8C-000423D98634.root', 
        '/store/relval/CMSSW_3_3_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP31X_V4-v1/0012/3E5C88E4-DC95-DE11-8694-000423D9863C.root')
)


process.saveTauEff = cms.EDAnalyzer("DQMSimpleFileSaver",
    outputFileName = cms.string('/afs/cern.ch/user/g/gennai/scratch.0/Validation/CMSSW_3_3_3/src/Validation/RecoTau/test/TauID/QCD_recoFiles/TauVal_CMSSW_3_3_3_QCD.root')
)



process.TauEfficiencies = cms.EDAnalyzer("DQMHistEffProducer",
    plots = cms.PSet(
        PFTauHighEfficiencyLeadingPionIDMatchingEfficiencies = cms.PSet(
            efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
            denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible')
        ),
        PFTauHighEfficiencyLeadingPionIDECALIsolationEfficienies = cms.PSet(
            efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
            denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolation/shrinkingConePFTauDiscriminationByECALIsolation_vs_#PAR#TauVisible')
        ),
        PFTauHighEfficiencyLeadingPionIDTrackIsolationEfficienies = cms.PSet(
            efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
            denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolation/shrinkingConePFTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible')
        ),
        PFTauHighEfficiencyLeadingPionIDLeadingPionPtCutEfficiencies = cms.PSet(
            efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
            denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/shrinkingConePFTauDiscriminationByLeadingPionPtCut_vs_#PAR#TauVisible')
        ),
 PFTauHighEfficiencyLeadingPionIDLeadingTrackPtCutEfficiencies = cms.PSet(
            efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
            denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
            parameter = cms.vstring('pt', 
                'eta', 
                'phi', 
                'energy'),
            numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/shrinkingConePFTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible')
        )
    )
)
process.kinematicSelectedTauValDenominator = cms.EDFilter("PFTauSelector",
     src = cms.InputTag("shrinkingConePFTauProducer"),
 discriminators = cms.VPSet(
       cms.PSet( discriminator=cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),selectionCut=cms.double(-1.0))
    ),
     cut = cms.string('pt > 10. && abs(eta) < 2.5'),
     filter = cms.bool(False)
)



process.PFTausHighEfficiencyLeadingPionBothProngs = cms.EDAnalyzer("TauTagValidation",
    MatchDeltaR_Leptons = cms.double(0.15),
    SaveOutputHistograms = cms.bool(False),
    RefCollection = cms.InputTag("kinematicSelectedTauValDenominator"),
    MatchDeltaR_Jets = cms.double(0.3),
    DataType = cms.string('Leptons'),
    discriminators = cms.VPSet(cms.PSet(
        discriminator = cms.string('shrinkingConePFTauDiscriminationByLeadingPionPtCut'),
        selectionCut = cms.double(0.5)
    ), 
        cms.PSet(
            discriminator = cms.string('shrinkingConePFTauDiscriminationByLeadingTrackPtCut'),
            selectionCut = cms.double(0.5)
    ), 
        cms.PSet(
            discriminator = cms.string('shrinkingConePFTauDiscriminationByTrackIsolation'),
            selectionCut = cms.double(0.5)
        ), 
        cms.PSet(
            discriminator = cms.string('shrinkingConePFTauDiscriminationByECALIsolation'),
            selectionCut = cms.double(0.5)
        )),
    TauProducer = cms.InputTag("shrinkingConePFTauProducer"),
    ExtensionName = cms.string('LeadingPion')
)


process.TauValNumeratorAndDenominator = cms.Sequence(process.PFTausHighEfficiencyLeadingPionBothProngs)




process.runTauValidationBatchMode = cms.Sequence(process.TauValNumeratorAndDenominator)

process.runTauValidation = cms.Sequence(process.runTauValidationBatchMode+process.TauEfficiencies)


process.validation = cms.Path(process.kinematicSelectedTauValDenominator*process.runTauValidation*process.saveTauEff)


process.DQMStore = cms.Service("DQMStore")



process.StandardMatchingParameters = cms.PSet(
    MatchDeltaR_Leptons = cms.double(0.15),
    SaveOutputHistograms = cms.bool(False),
    MatchDeltaR_Jets = cms.double(0.3),
    RefCollection = cms.InputTag("kinematicSelectedTauValDenominator"),
    DataType = cms.string('Leptons')
)



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.schedule = cms.Schedule(process.validation)

