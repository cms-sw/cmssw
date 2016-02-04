import FWCore.ParameterSet.Config as cms
import copy
import re
import os

"""

   RecoTauValidation_cfi.py

   Contains the standard tau validation parameters.  It is organized into
   the following sections.

   DENOMINATOR 
     
     Set common kinematic cuts (pt > 5 and eta < 2.5) on the denominator source.
     Note that the denominator depends on the type of test (signal/background/e etc)

     The denominator kinematic cutter requires that 

   HISTOGRAMS

     Produce numerator and denominator histgorams used to produce
     tau efficiency plots

        Provides sequence: 
          TauValNumeratorAndDenominator 
        Requires:
          tauValSelectedDenominator (filtered GenJet collection)
        
   EFFICIENCY
   
     Using numerator and denominators, calculate and store
     the efficiency curves

        Provides sequence:
          TauEfficiencies
        Requires:
          TauValNumeratorAndDenominator

   PLOTTING

     Plot curves calculated in efficiency, in both an overlay mode
     showing overall performance for a release, and the indvidual 
     discriminator efficiency compared to a given release

        Provides sequence:
          loadTau
          plotTauValidation
          loadAndPlotTauValidation

        Requires:
          TauEfficiencies, external root file to compare to

     Plotting must be executed in a separate cmsRun job!

   UTILITIES
     
     Various scripts to automate things...


"""

"""

DENOMINATOR

"""

# require generator level hadrons produced in tau-decay to have transverse momentum above threshold
kinematicSelectedTauValDenominator = cms.EDFilter("GenJetSelector",
     src = cms.InputTag("objectTypeSelectedTauValDenominator"),
     cut = cms.string('pt > 5. && abs(eta) < 2.5'),
     filter = cms.bool(False)
)


"""

HISTOGRAMS

        Plot the pt/eta/energy/phi spectrum of PFTaus that pass 
        a series of PFTauDiscriminator cuts.

        These will be used as the numerator/denominators of the
        efficiency calculations
"""

StandardMatchingParameters = cms.PSet(
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   #RefCollection                = cms.InputTag("TauGenJetProducer","selectedGenTauDecaysToHadronsPt5Cumulative"),
   RefCollection                = cms.InputTag("kinematicSelectedTauValDenominator"),
)

PFTausHighEfficiencyLeadingPionBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string("LeadingPion"),
   TauProducer                  = cms.InputTag('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

PFTausHighEfficiencyBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.InputTag('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

RunTancValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunTancValidation.ExtensionName = "Tanc"
RunTancValidation.discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
)

RunHPSValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunHPSValidation.ExtensionName = ""
RunHPSValidation.TauProducer   = cms.InputTag('hpsPFTauProducer')
RunHPSValidation.discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByLooseIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByMediumIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationByTightIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("hpsPFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
)



PFTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.InputTag('fixedConePFTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

CaloTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardMatchingParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.InputTag('caloRecoTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5))
 )
)

TauValNumeratorAndDenominator = cms.Sequence(
      PFTausBothProngs+
      CaloTausBothProngs +
      PFTausHighEfficiencyBothProngs+
      PFTausHighEfficiencyLeadingPionBothProngs+
      RunTancValidation+
      RunHPSValidation
      )

"""

EFFICIENCY

        Tau efficiency calculations

        Define the Efficiency curves to produce.  Each
        efficiency producer takes the numberator and denominator
        histograms and the dependent variables.
"""

TauEfficiencies = cms.EDAnalyzer("DQMHistEffProducer",
    plots = cms.PSet(
# REGULAR PFTAU EFFICIENCIES CALCULATION
      PFTauIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_Matched/fixedConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackFinding/fixedConePFTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/fixedConePFTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/fixedConePFTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/fixedConePFTauDiscriminationByECALIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/fixedConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/fixedConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/fixedConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
#HPS EfficiencyCalculation
      HPSIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_Matched/hpsPFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDDecayModeFindingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDLooseIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByLooseIsolation/hpsPFTauDiscriminationByLooseIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDMediumIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByMediumIsolation/hpsPFTauDiscriminationByMediumIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByMediumIsolation/MediumIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDTightIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByTightIsolation/hpsPFTauDiscriminationByTightIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByTightIsolation/TightIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstElectron/hpsPFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      HPSIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstMuon/hpsPFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/hpsPFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      



# PFTAUHIGHEFFICIENCY EFFICIENCY CALCULATION
      PFTauHighEfficiencyIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackFinding/shrinkingConePFTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/shrinkingConePFTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/shrinkingConePFTauDiscriminationByTrackIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/shrinkingConePFTauDiscriminationByECALIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/shrinkingConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/shrinkingConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
# PFTAUHIGHEFFICIENCY_LEADING_PION EFFICIENCY CALCULATION
      PFTauHighEfficiencyLeadingPionIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDLeadingPionPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/shrinkingConePFTauDiscriminationByLeadingPionPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDTrackIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDECALIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/shrinkingConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      PFTauHighEfficiencyLeadingPionIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/shrinkingConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),      
# Tanc efficiency calculations
      ShrinkingConeTancIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_Matched/shrinkingConePFTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_Matched/PFJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDLeadingPionPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/shrinkingConePFTauDiscriminationByLeadingPionPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDOnePercentEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/shrinkingConePFTauDiscriminationByTaNCfrOnePercent_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/TaNCfrOnePercentEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDHalfPercentEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/shrinkingConePFTauDiscriminationByTaNCfrHalfPercent_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/TaNCfrHalfPercentEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDQuarterPercentEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/TaNCfrQuarterPercentEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDTenthPercentEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/shrinkingConePFTauDiscriminationByTaNCfrTenthPercent_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/TaNCfrTenthPercentEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/shrinkingConePFTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      ShrinkingConeTancIDElectronRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/shrinkingConePFTauDiscriminationAgainstMuon_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),      
# CALOTAU EFFICIENCY CALCULATIONS      
      CaloTauIDMatchingEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_Matched/caloRecoTauProducerMatched_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDLeadingTrackFindEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackFinding/caloRecoTauDiscriminationByLeadingTrackFinding_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackFinding/LeadingTrackFindingEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDLeadingTrackPtCutEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/caloRecoTauDiscriminationByLeadingTrackPtCut_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDIsolationEfficienies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/caloRecoTauDiscriminationByIsolation_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      ),
      CaloTauIDMuonRejectionEfficiencies = cms.PSet(
        numerator = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/caloRecoTauDiscriminationAgainstElectron_vs_#PAR#TauVisible'),
        denominator = cms.string('RecoTauV/caloRecoTauProducer_ReferenceCollection/nRef_Taus_vs_#PAR#TauVisible'),
        efficiency = cms.string('RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
        parameter = cms.vstring('pt', 'eta', 'phi', 'energy')
      )      
    )                                
)


 
"""

PLOTTING

        loadTau:  load two separate TauVal root files into the DQM
                  so the plotter can access them

"""

loadTau = cms.EDAnalyzer("DQMFileLoader",
  test = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/f/friis/scratch0/MyValidationArea/310pre6NewTags/src/Validation/RecoTau/test/CMSSW_3_1_0_pre6_ZTT_0505Fixes.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('test')
  ),
  reference = cms.PSet(
    inputFileNames = cms.vstring('/afs/cern.ch/user/f/friis/CMSSW/pre6/src/Validation/RecoTau/test/CMSSW_3_1_0_pre6_ZTT.root'),
    scaleFactor = cms.double(1.),
    dqmDirectory_store = cms.string('reference')
  )
)

# Lots of junk to define the plot style

# standard drawing stuff
standardDrawingStuff = cms.PSet(
  canvasSizeX = cms.int32(640),
  canvasSizeY = cms.int32(640),                         
  indOutputFileName = cms.string('#PLOT#.png'),
  xAxes = cms.PSet(
    pt = cms.PSet(
      xAxisTitle = cms.string('P_{T} / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    eta = cms.PSet(
      xAxisTitle = cms.string('#eta'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    phi = cms.PSet(
      xAxisTitle = cms.string('#phi'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    ),
    energy = cms.PSet(
      xAxisTitle = cms.string('E / GeV'),
      xAxisTitleOffset = cms.double(0.9),
      xAxisTitleSize = cms.double(0.05)
    )
  ),

  yAxes = cms.PSet(                         
    efficiency = cms.PSet(
      yScale = cms.string('linear'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      minY_log = cms.double(0.001),
      maxY_log = cms.double(1.8),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    ),
    fakeRate = cms.PSet(
      yScale = cms.string('log'), # linear/log
      minY_linear = cms.double(0.),
      maxY_linear = cms.double(1.6),
      minY_log = cms.double(0.001),
      maxY_log = cms.double(1.8),
      yAxisTitle = cms.string('#varepsilon'), 
      yAxisTitleOffset = cms.double(1.1),
      yAxisTitleSize = cms.double(0.05)
    )
  ),

  legends = cms.PSet(
    efficiency = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.72),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.17),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    ),
    efficiency_overlay = cms.PSet(
      posX = cms.double(0.50),
      posY = cms.double(0.66),
      sizeX = cms.double(0.39),
      sizeY = cms.double(0.23),
      header = cms.string(''),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0)
    )
  ),

  labels = cms.PSet(
    pt = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.77),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('P_{T} > 5 GeV')
    ),
    eta = cms.PSet(
      posX = cms.double(0.19),
      posY = cms.double(0.83),
      sizeX = cms.double(0.12),
      sizeY = cms.double(0.04),
      option = cms.string('brNDC'),
      borderSize = cms.int32(0),
      fillColor = cms.int32(0),
      textColor = cms.int32(1),
      textSize = cms.double(0.04),
      textAlign = cms.int32(22),
      text = cms.vstring('-2.5 < #eta < +2.5')
    )
  ),

  drawOptionSets = cms.PSet(
    efficiency = cms.PSet(
      test = cms.PSet(
        markerColor = cms.int32(4),
        markerSize = cms.double(1.),
        markerStyle = cms.int32(20),
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        drawOption = cms.string('ep'),
        drawOptionLegend = cms.string('p')
      ),
      reference = cms.PSet(
        lineColor = cms.int32(1),
        lineStyle = cms.int32(1),
        lineWidth = cms.int32(1),
        fillColor = cms.int32(41),
        drawOption = cms.string('eBand'),
        drawOptionLegend = cms.string('l')
      )
    )
  ),
                                     
  drawOptionEntries = cms.PSet(
    eff_overlay01 = cms.PSet(
      markerColor = cms.int32(1),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(1),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay02 = cms.PSet(
      markerColor = cms.int32(2),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(2),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay03 = cms.PSet(
      markerColor = cms.int32(3),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(3),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay04 = cms.PSet(
      markerColor = cms.int32(4),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(4),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay05 = cms.PSet(
      markerColor = cms.int32(6),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(6),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    ),
    eff_overlay06 = cms.PSet(
      markerColor = cms.int32(5),
      markerSize = cms.double(1.),
      markerStyle = cms.int32(20),
      lineColor = cms.int32(5),
      lineStyle = cms.int32(1),
      lineWidth = cms.int32(2),
      drawOption = cms.string('ex0'),
      drawOptionLegend = cms.string('p')
    )
  ),
)

standardCompareTestAndReference = cms.PSet(
  processes = cms.PSet(
    test = cms.PSet(
      dqmDirectory = cms.string('test'),
      legendEntry = cms.string('no test label'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    ),
    reference = cms.PSet(
      dqmDirectory = cms.string('reference'),
      legendEntry = cms.string('no ref label'),
      type = cms.string('smMC') # Data/smMC/bsmMC/smSumMC
    )
  ),
)

standardEfficiencyParameters = cms.PSet(
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency'),
      labels = cms.vstring('pt', 'eta'),
      drawOptionSet = cms.string('efficiency')
)

standardEfficiencyOverlay = cms.PSet(
      parameter = cms.vstring('pt', 'eta', 'phi', 'energy'),
      title = cms.string('TauId step by step efficiencies'),
      xAxis = cms.string('#PAR#'),
      yAxis = cms.string('efficiency'),
      legend = cms.string('efficiency_overlay'),
      labels = cms.vstring('pt', 'eta')
)

        
##################################################
#
#   The plotting of all the PFTau ID efficiencies
#
##################################################
plotPFTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      )
    ),
    LeadingTrackPtCutEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      )  
    ),
    TrackIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    ECALIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstMuonEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/fixedConePFTauProducer_fixedConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
    )
  ),
  outputFilePath = cms.string('./fixedConePFTauProducer/'),
)                    
##################################################
#
#   The plotting of HPS Efficiencies
#
##################################################


plotHPSEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    DecayModeEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
    ),
    LooseIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    MediumIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByMediumIsolation/MediumIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TightIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByTightIsolation/TightIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstMuonEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Decay Mode Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Loose Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByMediumIsolation/MediumIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Medium Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/hpsPFTauProducer_hpsPFTauDiscriminationByTightIsolation/TightIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Tight Iso.')
        )
      )
    )
  ),
  outputFilePath = cms.string('./hpsPFTauProducer/')
)      






##################################################
#
#   The plotting of all the PFTauHighEfficiencies ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    LeadingTrackPtCutEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
    ),
    TrackIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    ECALIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),    
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstMuonEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByTrackIsolation/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationByECALIsolation/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso.')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducer_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducer/'),
)      

##################################################
#
#   The plotting of all the CaloTau ID efficiencies
#
##################################################

plotCaloTauEfficiencies = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    CaloJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    LeadingTrackPtCutEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
    ),
    IsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_Matched/CaloJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('CaloJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByLeadingTrackPtCut/LeadingTrackPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Track Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationByIsolation/IsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Isolation')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/caloRecoTauProducer_caloRecoTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Electron Rejection')
        )
      ),
    )
  ),
  outputFilePath = cms.string('./caloRecoTauProducer/'),
)

##################################################
#
#   Plot Tanc performance
#
##################################################

plotTancValidation = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    LeadingTrackPtCutEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
    ),
    TaNCfrOnePercentEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/TaNCfrOnePercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TaNCfrHalfPercentEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/TaNCfrHalfPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TaNCfrQuarterPercentEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/TaNCfrQuarterPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TaNCfrTenthPercentEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/TaNCfrTenthPercentEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstMuonEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Pion Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrOnePercent/TaNCfrOnePercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC One Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrHalfPercent/TaNCfrHalfPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Half Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent/TaNCfrQuarterPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Quarter Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationByTaNCfrTenthPercent/TaNCfrTenthPercentEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('TaNC Tenth Percent')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerTanc_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducerTanc/'),
)


##################################################
#
#   The plotting of all the PFTauHighEfficiencyUsingLeadingPion ID efficiencies
#
##################################################

plotPFTauHighEfficiencyEfficienciesLeadingPion = cms.EDAnalyzer("DQMHistPlotter",
  standardDrawingStuff,
  standardCompareTestAndReference,
  drawJobs = cms.PSet(                                     
    PFJetMatchingEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    LeadingTrackPtCutEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),  
    ),
    TrackIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    ECALIsolationEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstElectronEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    AgainstMuonEff = cms.PSet(
      standardEfficiencyParameters,
      plots = cms.PSet(
        dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'), 
        processes = cms.vstring('test', 'reference')
      ),
    ),
    TauIdEffStepByStep = cms.PSet(
      standardEfficiencyOverlay,
      plots = cms.VPSet(
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_Matched/PFJetMatchingEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay01'),
          legendEntry = cms.string('PFJet Matching')
        ),    
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByLeadingPionPtCut/LeadingPionPtCutEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay02'),
          legendEntry = cms.string('Lead Pion Finding')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion/TrackIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay03'),
          legendEntry = cms.string('Track Iso. Using Lead. Pion')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion/ECALIsolationEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay04'),
          legendEntry = cms.string('Track + Gamma Iso. Using Lead. Pioon')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstElectron/AgainstElectronEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay05'),
          legendEntry = cms.string('Electron Rejection')
        ),
        cms.PSet(
          dqmMonitorElements = cms.vstring('#PROCESSDIR#/RecoTauV/shrinkingConePFTauProducerLeadingPion_shrinkingConePFTauDiscriminationAgainstMuon/AgainstMuonEff#PAR#'),
          process = cms.string('test'),
          drawOptionEntry = cms.string('eff_overlay06'),
          legendEntry = cms.string('Muon Rejection')
        )
      ),
    )
  ),
  outputFilePath = cms.string('./shrinkingConePFTauProducerLeadingPion/'),
)

plotTauValidation = cms.Sequence(
      plotPFTauEfficiencies
      +plotPFTauHighEfficiencyEfficiencies
      +plotCaloTauEfficiencies
      +plotTancValidation
      +plotPFTauHighEfficiencyEfficienciesLeadingPion
      +plotHPSEfficiencies
      )


loadAndPlotTauValidation = cms.Sequence(
      loadTau
      +plotTancValidation
      )

"""

UTILITIES

"""

class ApplyFunctionToSequence:
   """ Helper class that applies a given function to all modules
       in a sequence """
   def __init__(self,function):
      self.functor = function
   def enter(self, module):
      self.functor(module)
   def leave(self, module):
      pass

def TranslateToLegacyProdNames(input):
   input = re.sub('fixedConePFTauProducer', 'pfRecoTauProducer', input)
   #fixedDiscriminationRegex = re.compile('fixedConePFTauDiscrimination( \w* )')
   fixedDiscriminationRegex = re.compile('fixedConePFTauDiscrimination(\w*)')
   input = fixedDiscriminationRegex.sub(r'pfRecoTauDiscrimination\1', input)
   input = re.sub('shrinkingConePFTauProducer', 'pfRecoTauProducerHighEfficiency', input)
   shrinkingDiscriminationRegex = re.compile('shrinkingConePFTauDiscrimination(\w*)')
   input = shrinkingDiscriminationRegex.sub(r'pfRecoTauDiscrimination\1HighEfficiency', input)
   return input


def ConvertDrawJobToLegacyCompare(input):
   """ Converts a draw job defined to compare 31X named PFTau validtion efficiencies
       to comapre a 31X to a 22X named validation """
   # get the list of drawjobs { name : copyOfPSet }
   if not hasattr(input, "drawJobs"):
      return
   myDrawJobs = input.drawJobs.parameters_()
   for drawJobName, drawJobData in myDrawJobs.iteritems():
      print drawJobData
      if not drawJobData.plots.pythonTypeName() == "cms.PSet":
         continue
      pSetToInsert = cms.PSet(
            standardEfficiencyParameters,
            plots = cms.VPSet(
               # test plot w/ modern names
               cms.PSet(
                  dqmMonitorElements = drawJobData.plots.dqmMonitorElements,
                  process = cms.string('test'),
                  drawOptionEntry = cms.string('eff_overlay01'),
                  legendEntry = cms.string(input.processes.test.legendEntry.value())
                  ),
               # ref plot w/ vintage name
               cms.PSet(
                  # translate the name
                  dqmMonitorElements = cms.vstring(TranslateToLegacyProdNames(drawJobData.plots.dqmMonitorElements.value()[0])),
                  process = cms.string('reference'),
                  drawOptionEntry = cms.string('eff_overlay02'),
                  legendEntry = cms.string(input.processes.reference.legendEntry.value())
                  )
               )
            )
      input.drawJobs.__setattr__(drawJobName, pSetToInsert)

def MakeLabeler(TestLabel, ReferenceLabel):
   def labeler(module):
      if hasattr(module, 'processes'):
         if module.processes.hasParameter(['test', 'legendEntry']) and module.processes.hasParameter([ 'reference', 'legendEntry']):
            module.processes.test.legendEntry = TestLabel
            module.processes.reference.legendEntry = ReferenceLabel
            print "Set test label to %s and reference label to %s for plot producer %s" % (TestLabel, ReferenceLabel, module.label())
         else:
            print "ERROR in RecoTauValidation_cfi::MakeLabeler - trying to set test/reference label but %s does not have processes.(test/reference).legendEntry parameters!" % module.label()
   return labeler

def SetYmodulesToLog(matchingNames = []):
   ''' set all modules whose name contains one of the matching names to log y scale'''
   def yLogger(module):
      ''' set a module to use log scaling in the yAxis'''
      if hasattr(module, 'drawJobs'):
         print "EK DEBUG"
         drawJobParamGetter = lambda subName : getattr(module.drawJobs, subName)
         #for subModule in [getattr(module.drawJobs, subModuleName) for subModuleName in dir(module.drawJobs)]:
         attrNames = dir(module.drawJobs)
         for subModuleName, subModule in zip(attrNames, map(drawJobParamGetter, attrNames)):
            matchedNames = [name for name in matchingNames if subModuleName.find( name) > -1] # matching sub strings
            if hasattr(subModule, "yAxis") and len(matchedNames):
               print "Setting drawJob: ", subModuleName, " to log scale."
               subModule.yAxis = cms.string('fakeRate') #'fakeRate' configuration specifies the log scaling
   return yLogger


def SetBaseDirectory(Directory):
   def BaseDirectorizer(module):
      newPath = Directory
      #if module.hasParameter("outputFilePath"):
      if hasattr(module, "outputFilePath"):
         oldPath = module.outputFilePath.value()
         newPath = os.path.join(newPath, oldPath)
         if not os.path.exists(newPath):
            os.makedirs(newPath)
         print newPath
         module.outputFilePath = cms.string("%s" % newPath)
   return BaseDirectorizer

def RemoveComparisonPlotCommands(module):
   if hasattr(module, 'drawJobs'):
      #get draw job parameter names
      drawJobs = module.drawJobs.parameterNames_()
      for drawJob in drawJobs:
         if drawJob != "TauIdEffStepByStep":
            module.drawJobs.__delattr__(drawJob)
            print "Removing comparison plot", drawJob

def SetPlotDirectory(myPlottingSequence, directory):
   myFunctor = ApplyFunctionToSequence(SetBaseDirectory(directory))
   myPlottingSequence.visit(myFunctor)

def SetTestAndReferenceLabels(myPlottingSequence, TestLabel, ReferenceLabel):
   myFunctor = ApplyFunctionToSequence(MakeLabeler(TestLabel, ReferenceLabel))
   myPlottingSequence.visit(myFunctor)

def SetCompareToLegacyProductNames(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(ConvertDrawJobToLegacyCompare)
   myPlottingSequence.visit(myFunctor)

def SetTestFileToPlot(myProcess, FileLoc):
   myProcess.loadTau.test.inputFileNames = cms.vstring(FileLoc)

def SetReferenceFileToPlot(myProcess, FileLoc):
   if FileLoc == None:
      del myProcess.loadTau.reference
   else:
      myProcess.loadTau.reference.inputFileNames = cms.vstring(FileLoc)

def SetLogScale(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(SetYmodulesToLog())
   myPlottingSequence.visit(myFunctor)

def SetSmartLogScale(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(SetYmodulesToLog(['Electron', 'Muon', 'Isolation', 'TaNC']))
   myPlottingSequence.visit(myFunctor)

def SetPlotOnlyStepByStep(myPlottingSequence):
   myFunctor = ApplyFunctionToSequence(RemoveComparisonPlotCommands)
   myPlottingSequence.visit(myFunctor)
