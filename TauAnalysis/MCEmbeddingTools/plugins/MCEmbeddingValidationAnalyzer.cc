#include "TauAnalysis/MCEmbeddingTools/plugins/MCEmbeddingValidationAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"if ( genTau1 ) 

MCEmbeddingValidationAnalyzer::MCEmbeddingValidationAnalyzer(const edm::ParameterSet& cfg)
  : srcReplacedMuons_(cfg.getParameter<edm::InputTag>("srcReplacedMuons")),
    srcRecMuons_(cfg.getParameter<edm::InputTag>("srcRecMuons")),
    srcRecTracks_(cfg.getParameter<edm::InputTag>("srcRecTracks")),
    srcCaloTowers_(cfg.getParameter<edm::InputTag>("srcCaloTowers")),
    srcRecPFCandidates_(cfg.getParameter<edm::InputTag>("srcRecPFCandidates")),
    srcRecJets_(cfg.getParameter<edm::InputTag>("srcRecJets")),
    srcRecVertex_(cfg.getParameter<edm::InputTag>("srcRecVertex")),
    srcGenDiTaus_(cfg.getParameter<edm::InputTag>("srcGenDiTaus")),
    srcGenLeg1_(cfg.getParameter<edm::InputTag>("srcGenLeg1")),
    srcRecLeg1_(cfg.getParameter<edm::InputTag>("srcRecLeg1")),
    srcGenLeg2_(cfg.getParameter<edm::InputTag>("srcGenLeg2")),
    srcRecLeg2_(cfg.getParameter<edm::InputTag>("srcRecLeg2")),
    srcGenParticles_(cfg.getParameter<edm::InputTag>("srcGenParticles")),
    srcL1ETM_(cfg.getParameter<edm::InputTag>("srcL1ETM")),
    srcGenMEt_(cfg.getParameter<edm::InputTag>("srcGenMEt")),
    srcRecCaloMEt_(cfg.getParameter<edm::InputTag>("srcRecCaloMEt")),
    srcWeights_(cfg.getParameter<vInputTag>("srcWeights")),
    srcGenFilterInfo_(cfg.getParameter<edm::InputTag>("srcGenFilterInfo")),
    dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")),
    replacedMuonPtThresholdHigh_(cfg.getParameter<double>("replacedMuonPtThresholdHigh")),
    replacedMuonPtThresholdLow_(cfg.getParameter<double>("replacedMuonPtThresholdLow"))
{
  typedef std::pair<int, int> pint;
  std::vector<pint> jetBins;
  jetBins.push_back(pint(-1, -1));
  jetBins.push_back(pint(0, 0));
  jetBins.push_back(pint(1, 1));
  jetBins.push_back(pint(2, 2));
  jetBins.push_back(pint(3, 1000));
  for ( std::vector<pint>::const_iterator jetBin = jetBins.begin();
	jetBin != jetBins.end(); ++jetBin ) {
//--- setup electron Pt, eta and phi distributions;
//    electron id & isolation and trigger efficiencies
    setupLeptonDistribution(jetBin->first, jetBin->second, cfg, "electronDistributions", electronDistributions_);
    setupLeptonEfficiency(jetBin->first, jetBin->second, cfg, "electronEfficiencies", electronEfficiencies_);
    setupLeptonL1TriggerEfficiency(jetBin->first, jetBin->second, cfg, "electronL1TriggerEfficiencies", electronL1TriggerEfficiencies_);

//--- setup muon Pt, eta and phi distributions;
//    muon id & isolation and trigger efficiencies
    setupLeptonDistribution(jetBin->first, jetBin->second, cfg, "muonDistributions", muonDistributions_);
    setupLeptonEfficiency(jetBin->first, jetBin->second, cfg, "muonEfficiencies", muonEfficiencies_);
    setupLeptonL1TriggerEfficiency(jetBin->first, jetBin->second, cfg, "muonL1TriggerEfficiencies", muonL1TriggerEfficiencies_);
  
//--- setup tau Pt, eta and phi distributions;
//    tau id efficiency
    setupLeptonDistribution(jetBin->first, jetBin->second, cfg, "tauDistributions", tauDistributions_);
    setupTauDistributionExtra(jetBin->first, jetBin->second, cfg, "tauDistributions", tauDistributionsExtra_);
    setupLeptonEfficiency(jetBin->first, jetBin->second, cfg, "tauEfficiencies", tauEfficiencies_);

//--- setup Pt, eta and phi distributions of L1Extra objects
//   (electrons, muons, tau-jet, central and forward jets)
    setupL1ExtraObjectDistribution(jetBin->first, jetBin->second, cfg, "l1ElectronDistributions", l1ElectronDistributions_);
    setupL1ExtraObjectDistribution(jetBin->first, jetBin->second, cfg, "l1MuonDistributions", l1MuonDistributions_);
    setupL1ExtraObjectDistribution(jetBin->first, jetBin->second, cfg, "l1TauDistributions", l1TauDistributions_);
    setupL1ExtraObjectDistribution(jetBin->first, jetBin->second, cfg, "l1CentralJetDistributions", l1CentralJetDistributions_);
    setupL1ExtraObjectDistribution(jetBin->first, jetBin->second, cfg, "l1ForwardJetDistributions", l1ForwardJetDistributions_);

//--- setup MET Pt and phi distributions;
//    efficiency of L1 (Calo)MET trigger requirement
    setupMEtDistribution(jetBin->first, jetBin->second, cfg, "metDistributions", metDistributions_);
    setupMEtL1TriggerEfficiency(jetBin->first, jetBin->second, cfg, "metL1TriggerEfficiencies", metL1TriggerEfficiencies_);
  }
}

MCEmbeddingValidationAnalyzer::~MCEmbeddingValidationAnalyzer()
{
  for ( std::vector<plotEntryTypeL1ETM*>::iterator it = l1ETMplotEntries_.begin();
	it != l1ETMplotEntries_.end(); ++it ) {
    delete (*it);
  }

  cleanCollection(electronDistributions_);
  cleanCollection(electronEfficiencies_);
  cleanCollection(electronL1TriggerEfficiencies_);
  cleanCollection(muonDistributions_);
  cleanCollection(muonEfficiencies_);
  cleanCollection(muonL1TriggerEfficiencies_);
  cleanCollection(tauDistributions_);
  cleanCollection(tauDistributionsExtra_);
  cleanCollection(tauEfficiencies_);
  cleanCollection(l1ElectronDistributions_);
  cleanCollection(l1MuonDistributions_);
  cleanCollection(l1TauDistributions_);
  cleanCollection(l1CentralJetDistributions_);
  cleanCollection(l1ForwardJetDistributions_);
  cleanCollection(metDistributions_);
  cleanCollection(metL1TriggerEfficiencies_);
}

template <typename T>
void MCEmbeddingValidationAnalyzer::setupLeptonDistribution(int minJets, int maxJets,
							    const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonDistributionT<T>*>& leptonDistributions)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgLeptonDistributions = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgLeptonDistribution = cfgLeptonDistributions.begin();
	  cfgLeptonDistribution != cfgLeptonDistributions.end(); ++cfgLeptonDistribution ) {
      edm::InputTag srcGen = cfgLeptonDistribution->getParameter<edm::InputTag>("srcGen");
      std::string cutGen = cfgLeptonDistribution->exists("cutGen") ? 
	cfgLeptonDistribution->getParameter<std::string>("cutGen") : "";
      edm::InputTag srcRec = cfgLeptonDistribution->getParameter<edm::InputTag>("srcRec");
      std::string cutRec = cfgLeptonDistribution->exists("cutRec") ? 
	cfgLeptonDistribution->getParameter<std::string>("cutRec") : "";
      double dRmatch = cfgLeptonDistribution->exists("dRmatch") ? 
	cfgLeptonDistribution->getParameter<double>("dRmatch") : 0.3;
      std::string dqmDirectory = dqmDirectory_full(cfgLeptonDistribution->getParameter<std::string>("dqmDirectory"));
      leptonDistributionT<T>* leptonDistribution = new leptonDistributionT<T>(minJets, maxJets, srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      leptonDistributions.push_back(leptonDistribution);
    }
  }
}

void MCEmbeddingValidationAnalyzer::setupTauDistributionExtra(int minJets, int maxJets,
							      const edm::ParameterSet& cfg, const std::string& keyword, std::vector<tauDistributionExtra*>& tauDistributionsExtra)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgLeptonDistributions = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgLeptonDistribution = cfgLeptonDistributions.begin();
	  cfgLeptonDistribution != cfgLeptonDistributions.end(); ++cfgLeptonDistribution ) {
      edm::InputTag srcGen = cfgLeptonDistribution->getParameter<edm::InputTag>("srcGen");
      std::string cutGen = cfgLeptonDistribution->exists("cutGen") ? 
	cfgLeptonDistribution->getParameter<std::string>("cutGen") : "";
      edm::InputTag srcRec = cfgLeptonDistribution->getParameter<edm::InputTag>("srcRec");
      std::string cutRec = cfgLeptonDistribution->exists("cutRec") ? 
	cfgLeptonDistribution->getParameter<std::string>("cutRec") : "";
      double dRmatch = cfgLeptonDistribution->exists("dRmatch") ? 
	cfgLeptonDistribution->getParameter<double>("dRmatch") : 0.3;
      std::string dqmDirectory = dqmDirectory_full(cfgLeptonDistribution->getParameter<std::string>("dqmDirectory"));
      tauDistributionExtra* tauDistribution = new tauDistributionExtra(minJets, maxJets, srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      tauDistributionsExtra.push_back(tauDistribution);
    }
  }
}

template <typename T>
void MCEmbeddingValidationAnalyzer::setupLeptonEfficiency(int minJets, int maxJets,
							  const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonEfficiencyT<T>*>& leptonEfficiencies)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgLeptonEfficiencies = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgLeptonEfficiency = cfgLeptonEfficiencies.begin();
	  cfgLeptonEfficiency != cfgLeptonEfficiencies.end(); ++cfgLeptonEfficiency ) {
      edm::InputTag srcGen = cfgLeptonEfficiency->getParameter<edm::InputTag>("srcGen");
      std::string cutGen = cfgLeptonEfficiency->exists("cutGen") ? 
	cfgLeptonEfficiency->getParameter<std::string>("cutGen") : "";
      edm::InputTag srcRec = cfgLeptonEfficiency->getParameter<edm::InputTag>("srcRec");
      std::string cutRec = cfgLeptonEfficiency->exists("cutRec") ? 
	cfgLeptonEfficiency->getParameter<std::string>("cutRec") : "";
      double dRmatch = cfgLeptonEfficiency->exists("dRmatch") ? 
	cfgLeptonEfficiency->getParameter<double>("dRmatch") : 0.3;
      std::string dqmDirectory = dqmDirectory_full(cfgLeptonEfficiency->getParameter<std::string>("dqmDirectory"));
      leptonEfficiencyT<T>* leptonEfficiency = new leptonEfficiencyT<T>(minJets, maxJets, srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      leptonEfficiencies.push_back(leptonEfficiency);
    }
  }
}

template <typename T1, typename T2>
void MCEmbeddingValidationAnalyzer::setupLeptonL1TriggerEfficiency(int minJets, int maxJets,
								   const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonL1TriggerEfficiencyT1T2<T1,T2>*>& leptonL1TriggerEfficiencies)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgLeptonL1TriggerEfficiencies = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgLeptonL1TriggerEfficiency = cfgLeptonL1TriggerEfficiencies.begin();
	  cfgLeptonL1TriggerEfficiency != cfgLeptonL1TriggerEfficiencies.end(); ++cfgLeptonL1TriggerEfficiency ) {
      edm::InputTag srcRef = cfgLeptonL1TriggerEfficiency->getParameter<edm::InputTag>("srcRef");
      std::string cutRef = cfgLeptonL1TriggerEfficiency->exists("cutRef") ? 
	cfgLeptonL1TriggerEfficiency->getParameter<std::string>("cutRef") : "";
      edm::InputTag srcL1 = cfgLeptonL1TriggerEfficiency->getParameter<edm::InputTag>("srcL1");
      std::string cutL1 = cfgLeptonL1TriggerEfficiency->getParameter<std::string>("cutL1");
      double dRmatch = cfgLeptonL1TriggerEfficiency->exists("dRmatch") ? 
	cfgLeptonL1TriggerEfficiency->getParameter<double>("dRmatch") : 0.3;
      std::string dqmDirectory = dqmDirectory_full(cfgLeptonL1TriggerEfficiency->getParameter<std::string>("dqmDirectory"));
      leptonL1TriggerEfficiencyT1T2<T1,T2>* leptonL1TriggerEfficiency = new leptonL1TriggerEfficiencyT1T2<T1,T2>(minJets, maxJets, srcRef, cutRef, srcL1, cutL1, dRmatch, dqmDirectory);
      leptonL1TriggerEfficiencies.push_back(leptonL1TriggerEfficiency);
    }
  }
}

template <typename T>
void MCEmbeddingValidationAnalyzer::setupL1ExtraObjectDistribution(int minJets, int maxJets,
								   const edm::ParameterSet& cfg, const std::string& keyword, std::vector<l1ExtraObjectDistributionT<T>*>& l1ExtraObjectDistributions)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgL1ExtraObjectDistributions = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgL1ExtraObjectDistribution = cfgL1ExtraObjectDistributions.begin();
	  cfgL1ExtraObjectDistribution != cfgL1ExtraObjectDistributions.end(); ++cfgL1ExtraObjectDistribution ) {
      edm::InputTag src = cfgL1ExtraObjectDistribution->getParameter<edm::InputTag>("src");
      std::string cut = cfgL1ExtraObjectDistribution->exists("cut") ? 
	cfgL1ExtraObjectDistribution->getParameter<std::string>("cut") : "";
      std::string dqmDirectory = dqmDirectory_full(cfgL1ExtraObjectDistribution->getParameter<std::string>("dqmDirectory"));
      l1ExtraObjectDistributionT<T>* l1ExtraObjectDistribution = new l1ExtraObjectDistributionT<T>(minJets, maxJets, src, cut, dqmDirectory);
      l1ExtraObjectDistributions.push_back(l1ExtraObjectDistribution);
    }
  }
}

void MCEmbeddingValidationAnalyzer::setupMEtDistribution(int minJets, int maxJets,
							 const edm::ParameterSet& cfg, const std::string& keyword, std::vector<metDistributionType*>& metDistributions)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgMEtDistributions = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgMEtDistribution = cfgMEtDistributions.begin();
	  cfgMEtDistribution != cfgMEtDistributions.end(); ++cfgMEtDistribution ) {
      edm::InputTag srcGen = cfgMEtDistribution->getParameter<edm::InputTag>("srcGen");
      edm::InputTag srcRec = cfgMEtDistribution->getParameter<edm::InputTag>("srcRec");
      edm::InputTag srcGenZs = cfgMEtDistribution->getParameter<edm::InputTag>("srcGenZs");
      std::string dqmDirectory = dqmDirectory_full(cfgMEtDistribution->getParameter<std::string>("dqmDirectory"));
      metDistributionType* metDistribution = new metDistributionType(minJets, maxJets, srcGen, srcRec, srcGenZs, dqmDirectory);
      metDistributions.push_back(metDistribution);
    }
  }
}
    
void MCEmbeddingValidationAnalyzer::setupMEtL1TriggerEfficiency(int minJets, int maxJets,
								const edm::ParameterSet& cfg, const std::string& keyword, std::vector<metL1TriggerEfficiencyType*>& metL1TriggerEfficiencies)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgMEtL1TriggerEfficiencies = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgMEtL1TriggerEfficiency = cfgMEtL1TriggerEfficiencies.begin();
	  cfgMEtL1TriggerEfficiency != cfgMEtL1TriggerEfficiencies.end(); ++cfgMEtL1TriggerEfficiency ) {
      edm::InputTag srcRef = cfgMEtL1TriggerEfficiency->getParameter<edm::InputTag>("srcRef");
      edm::InputTag srcL1 = cfgMEtL1TriggerEfficiency->getParameter<edm::InputTag>("srcL1");
      double cutL1Et = cfgMEtL1TriggerEfficiency->getParameter<double>("cutL1Et");
      double cutL1Pt = cfgMEtL1TriggerEfficiency->getParameter<double>("cutL1Pt");
      std::string dqmDirectory = dqmDirectory_full(cfgMEtL1TriggerEfficiency->getParameter<std::string>("dqmDirectory"));
      metL1TriggerEfficiencyType* metL1TriggerEfficiency = new metL1TriggerEfficiencyType(minJets, maxJets, srcRef, srcL1, cutL1Et, cutL1Pt, dqmDirectory);
      metL1TriggerEfficiencies.push_back(metL1TriggerEfficiency);
    }
  }
}

void MCEmbeddingValidationAnalyzer::beginJob()
{
  if ( !edm::Service<DQMStore>().isAvailable() ) 
    throw cms::Exception("MuonRadiationAnalyzer::beginJob")
      << "Failed to access dqmStore !!\n";

  DQMStore& dqmStore = (*edm::Service<DQMStore>());

//--- book all histograms
  histogramGenFilterEfficiency_                = dqmStore.book1D("genFilterEfficiency",                "genFilterEfficiency",                 102,     -0.01,         1.01);

  histogramRotationAngleMatrix_                = dqmStore.book2D("rfRotationAngleMatrix",              "rfRotationAngleMatrix",                 2,     -0.5,          1.5, 2, -0.5, 1.5);

  histogramNumTracksPtGt5_                     = dqmStore.book1D("numTracksPtGt5",                     "numTracksPtGt5",                       50,     -0.5,         49.5);
  histogramNumTracksPtGt10_                    = dqmStore.book1D("numTracksPtGt10",                    "numTracksPtGt10",                      50,     -0.5,         49.5);
  histogramNumTracksPtGt20_                    = dqmStore.book1D("numTracksPtGt20",                    "numTracksPtGt20",                      50,     -0.5,         49.5);
  histogramNumTracksPtGt30_                    = dqmStore.book1D("numTracksPtGt30",                    "numTracksPtGt30",                      50,     -0.5,         49.5);
  histogramNumTracksPtGt40_                    = dqmStore.book1D("numTracksPtGt40",                    "numTracksPtGt40",                      50,     -0.5,         49.5);
      
  histogramNumGlobalMuons_                     = dqmStore.book1D("numGlobalMuons",                     "numGlobalMuons",                       20,     -0.5,         19.5);
  histogramNumStandAloneMuons_                 = dqmStore.book1D("numStandAloneMuons",                 "numStandAloneMuons",                   20,     -0.5,         19.5);
  histogramNumPFMuons_                         = dqmStore.book1D("numPFMuons",                         "numPFMuons",                           20,     -0.5,         19.5);

  histogramNumChargedPFCandsPtGt5_             = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt10_            = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt20_            = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt30_            = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt40_            = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",               50,     -0.5,         49.5);

  histogramNumNeutralPFCandsPtGt5_             = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt10_            = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt20_            = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt30_            = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",               50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt40_            = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",               50,     -0.5,         49.5);
    
  histogramNumJetsPtGt20_                      = dqmStore.book1D("numJetsPtGt20",                      "numJetsPtGt20",                        20,     -0.5,         19.5);
  histogramNumJetsPtGt20AbsEtaLt2_5_           = dqmStore.book1D("numJetsPtGt20AbsEtaLt2_5",           "numJetsPtGt20AbsEtaLt2_5",             20,     -0.5,         19.5);
  histogramNumJetsPtGt20AbsEta2_5to4_5_        = dqmStore.book1D("numJetsPtGt20AbsEta2_5to4_5",        "numJetsPtGt20AbsEta2_5to4_5",          20,     -0.5,         19.5);
  histogramNumJetsPtGt30_                      = dqmStore.book1D("numJetsPtGt30",                      "numJetsPtGt30",                        20,     -0.5,         19.5);
  histogramNumJetsPtGt30AbsEtaLt2_5_           = dqmStore.book1D("numJetsPtGt30AbsEtaLt2_5",           "numJetsPtGt30AbsEtaLt2_5",             20,     -0.5,         19.5);
  histogramNumJetsPtGt30AbsEta2_5to4_5_        = dqmStore.book1D("numJetsPtGt30AbsEta2_5to4_5",        "numJetsPtGt30AbsEta2_5to4_5",          20,     -0.5,         19.5);
  
  histogramRecVertexX_                         = dqmStore.book1D("recVertexX",                         "recVertexX",                         2000,     -1.,          +1.);
  histogramRecVertexY_                         = dqmStore.book1D("recVertexY",                         "recVertexY",                         2000,     -1.,          +1.);
  histogramRecVertexZ_                         = dqmStore.book1D("recVertexZ",                         "recVertexZ",                          500,    -25.,         +25.);
  
  histogramGenDiTauPt_                         = dqmStore.book1D("genDiTauPt",                         "genDiTauPt",                          250,      0.,         250.);
  histogramGenDiTauEta_                        = dqmStore.book1D("genDiTauEta",                        "genDiTauEta",                         198,     -9.9,         +9.9);
  histogramGenDiTauPhi_                        = dqmStore.book1D("genDiTauPhi",                        "genDiTauPhi",                          72, -TMath::Pi(), +TMath::Pi());
  histogramGenDiTauMass_                       = dqmStore.book1D("genDiTauMass",                       "genDiTauMass",                        500,      0.,         500.);

  histogramGenVisDiTauPt_                      = dqmStore.book1D("genVisDiTauPt",                      "genVisDiTauPt",                       250,      0.,         250.);
  histogramGenVisDiTauEta_                     = dqmStore.book1D("genVisDiTauEta",                     "genVisDiTauEta",                      198,     -9.9,         +9.9);
  histogramGenVisDiTauPhi_                     = dqmStore.book1D("genVisDiTauPhi",                     "genVisDiTauPhi",                       72, -TMath::Pi(), +TMath::Pi());
  histogramGenVisDiTauMass_                    = dqmStore.book1D("genVisDiTauMass",                    "genVisDiTauMass",                     500,      0.,         500.);

  histogramRecVisDiTauPt_                      = dqmStore.book1D("recVisDiTauPt",                      "recVisDiTauPt",                       250,      0.,         250.);
  histogramRecVisDiTauEta_                     = dqmStore.book1D("recVisDiTauEta",                     "recVisDiTauEta",                      198,     -9.9,         +9.9);
  histogramRecVisDiTauPhi_                     = dqmStore.book1D("recVisDiTauPhi",                     "recVisDiTauPhi",                       72, -TMath::Pi(), +TMath::Pi());
  histogramRecVisDiTauMass_                    = dqmStore.book1D("recVisDiTauMass",                    "recVisDiTauMass",                     500,      0.,         500.);

  histogramGenLeg1Pt_                          = dqmStore.book1D("genLeg1Pt",                          "genLeg1Pt",                           250,      0.,         250.);
  histogramGenLeg1Eta_                         = dqmStore.book1D("genLeg1Eta",                         "genLeg1Eta",                          198,     -9.9,         +9.9);
  histogramGenLeg1Phi_                         = dqmStore.book1D("genLeg1Phi",                         "genLeg1Phi",                           72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg1X_                           = dqmStore.book1D("genLeg1X",                           "genLeg1X",                            102,     -0.01,         1.01);
  histogramRecLeg1X_                           = dqmStore.book1D("recLeg1X",                           "recLeg1X",                            102,     -0.01,         1.01);
  histogramGenLeg2Pt_                          = dqmStore.book1D("genLeg2Pt",                          "genLeg2Pt",                           250,      0.,         250.);
  histogramGenLeg2Eta_                         = dqmStore.book1D("genLeg2Eta",                         "genLeg2Eta",                          198,     -9.9,         +9.9);
  histogramGenLeg2Phi_                         = dqmStore.book1D("genLeg2Phi",                         "genLeg2Phi",                           72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg2X_                           = dqmStore.book1D("genLeg2X",                           "genLeg2X",                            102,     -0.01,         1.01);
  histogramRecLeg2X_                           = dqmStore.book1D("recLeg2X",                           "recLeg2X",                            102,     -0.01,         1.01);

  histogramSumGenParticlePt_                   = dqmStore.book1D("sumGenParticlePt",                   "sumGenParticlePt",                    250,      0.,         250.);
  histogramSumGenParticlePt_charged_           = dqmStore.book1D("sumGenParticlePt_charged",           "sumGenParticlePt_charged",            250,      0.,         250.);
  histogramGenMEt_                             = dqmStore.book1D("genMEt",                             "genMEt",                              250,      0.,         250.);

  histogramRecTrackMEt_                        = dqmStore.book1D("recTrackMEt",                        "recTrackMEt",                         250,      0.,         250.);
  histogramRecTrackSumEt_                      = dqmStore.book1D("recTrackSumEt",                      "recTrackSumEt",                      2500,      0.,        2500.);
  histogramRecCaloMEtECAL_                     = dqmStore.book1D("recCaloMEtECAL",                     "recCaloMEtECAL",                      250,      0.,         250.);
  histogramRecCaloSumEtECAL_                   = dqmStore.book1D("recCaloSumEtECAL",                   "recCaloSumEtECAL",                   2500,      0.,        2500.);
  histogramRecCaloMEtHCAL_                     = dqmStore.book1D("recCaloMEtHCAL",                     "recCaloMEtHCAL",                      250,      0.,         250.);
  histogramRecCaloSumEtHCAL_                   = dqmStore.book1D("recCaloSumEtHCAL",                   "recCaloSumEtHCAL",                   2500,      0.,        2500.);
  histogramRecCaloMEtHF_                       = dqmStore.book1D("recCaloMEtHF",                       "recCaloMEtHF",                        250,      0.,         250.);
  histogramRecCaloSumEtHF_                     = dqmStore.book1D("recCaloSumEtHF",                     "recCaloSumEtHF",                     2500,      0.,        2500.);
  histogramRecCaloMEtHO_                       = dqmStore.book1D("recCaloMEtHO",                       "recCaloMEtHO",                        250,      0.,         250.);  
  histogramRecCaloSumEtHO_                     = dqmStore.book1D("recCaloSumEtHO",                     "recCaloSumEtHO",                     2500,      0.,        2500.);

  std::vector<std::string> genTauDecayModes;
  genTauDecayModes.push_back(std::string("")); // all tau decay modes
  genTauDecayModes.push_back(std::string("oneProng0Pi0"));
  genTauDecayModes.push_back(std::string("oneProng1Pi0"));
  genTauDecayModes.push_back(std::string("oneProng2Pi0"));
  genTauDecayModes.push_back(std::string("threeProng0Pi0"));
  genTauDecayModes.push_back(std::string("threeProng1Pi0"));
  for ( std::vector<std::string>::const_iterator genTauDecayMode = genTauDecayModes.begin();
	genTauDecayMode != genTauDecayModes.end(); ++genTauDecayMode ) {
    plotEntryTypeL1ETM* l1ETMplotEntry = new plotEntryTypeL1ETM(*genTauDecayMode, dqmDirectory_);
    l1ETMplotEntry->bookHistograms(dqmStore);
    l1ETMplotEntries_.push_back(l1ETMplotEntry);
  }

  bookHistograms(electronDistributions_, dqmStore);
  bookHistograms(electronEfficiencies_, dqmStore);
  bookHistograms(electronL1TriggerEfficiencies_, dqmStore);
  bookHistograms(muonDistributions_, dqmStore);
  bookHistograms(muonEfficiencies_, dqmStore);
  bookHistograms(muonL1TriggerEfficiencies_, dqmStore);
  bookHistograms(tauDistributions_, dqmStore);
  bookHistograms(tauDistributionsExtra_, dqmStore);
  bookHistograms(tauEfficiencies_, dqmStore);
  bookHistograms(l1ElectronDistributions_, dqmStore);
  bookHistograms(l1MuonDistributions_, dqmStore);
  bookHistograms(l1TauDistributions_, dqmStore);
  bookHistograms(l1CentralJetDistributions_, dqmStore);
  bookHistograms(l1ForwardJetDistributions_, dqmStore);
  bookHistograms(metDistributions_, dqmStore);
  bookHistograms(metL1TriggerEfficiencies_, dqmStore);
}

namespace
{
  void fillVisPtEtaPhiMassDistributions(const edm::Event& evt, 
					const edm::InputTag& srcLeg1, const edm::InputTag& srcLeg2, 
					MonitorElement* histogram_visDiTauPt, MonitorElement* histogram_visDiTauEta, MonitorElement* histogram_visDiTauPhi, MonitorElement* histogram_visDiTauMass,
					double evtWeight)
  {
    std::cout << "<fillVisPtEtaPhiMassDistributions>:" << std::endl;
    std::cout << " srcLeg1 = " << srcLeg1 << std::endl;
    std::cout << " srcLeg2 = " << srcLeg2 << std::endl;
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> visDecayProducts1;
    evt.getByLabel(srcLeg1, visDecayProducts1);
    edm::Handle<CandidateView> visDecayProducts2;
    evt.getByLabel(srcLeg2, visDecayProducts2);
    for ( edm::View<reco::Candidate>::const_iterator visDecayProduct1 = visDecayProducts1->begin();
	  visDecayProduct1 != visDecayProducts1->end(); ++visDecayProduct1 ) {
      for ( edm::View<reco::Candidate>::const_iterator visDecayProduct2 = visDecayProducts2->begin();
	  visDecayProduct2 != visDecayProducts2->end(); ++visDecayProduct2 ) {
	double dR = deltaR(visDecayProduct1->p4(), visDecayProduct2->p4());
	if ( dR > 0.3 ) { // protection in case srcLeg1 and srcLeg2 refer to same collection (e.g. both hadronic tau decays)
	  std::cout << "leg1: Pt = " << visDecayProduct1->pt() << ", phi = " << visDecayProduct1->phi() << " (Px = " << visDecayProduct1->px() << ", Py = " << visDecayProduct1->py() << ")" << std::endl;
	  std::cout << "leg2: Pt = " << visDecayProduct2->pt() << ", phi = " << visDecayProduct2->phi() << " (Px = " << visDecayProduct2->px() << ", Py = " << visDecayProduct2->py() << ")" << std::endl;
	  reco::Candidate::LorentzVector visDiTauP4 = visDecayProduct1->p4() + visDecayProduct2->p4();
	  histogram_visDiTauPt->Fill(visDiTauP4.pt(), evtWeight);
	  histogram_visDiTauEta->Fill(visDiTauP4.eta(), evtWeight);
	  histogram_visDiTauPhi->Fill(visDiTauP4.phi(), evtWeight);
	  histogram_visDiTauMass->Fill(visDiTauP4.mass(), evtWeight);
	}
      }
    }
  }

  void fillX1andX2Distributions(const edm::Event& evt, 
				const edm::InputTag& srcGenDiTau, const edm::InputTag& srcLeg1, const edm::InputTag& srcLeg2, 
				MonitorElement* histogram_leg1Pt, MonitorElement* histogram_leg1Eta, MonitorElement* histogram_leg1Phi, MonitorElement* histogram_leg1X, 
				MonitorElement* histogram_leg2Pt, MonitorElement* histogram_leg2Eta, MonitorElement* histogram_leg2Phi, MonitorElement* histogram_leg2X, 
				double evtWeight)
  {
    //std::cout << "<fillX1andX2Distributions>:" << std::endl;
    //std::cout << " srcLeg1 = " << srcLeg1.label() << std::endl;
    //std::cout << " srcLeg2 = " << srcLeg2.label() << std::endl;
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> genDiTaus;
    evt.getByLabel(srcGenDiTau, genDiTaus);
    edm::Handle<CandidateView> visDecayProducts1;    
    evt.getByLabel(srcLeg1, visDecayProducts1);
    //std::cout << "#visDecayProducts1 = " << visDecayProducts1->size() << std::endl;
    edm::Handle<CandidateView> visDecayProducts2;    
    evt.getByLabel(srcLeg2, visDecayProducts2);
    //std::cout << "#visDecayProducts2 = " << visDecayProducts2->size() << std::endl;
    for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	  genDiTau != genDiTaus->end(); ++genDiTau ) {
      const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
      if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) continue;
      const reco::Candidate* genLeg1 = genDiTau_composite->daughter(0);
      const reco::Candidate* genLeg2 = genDiTau_composite->daughter(1);
      //std::cout << "genLeg1: Pt = " << genLeg1->pt() << ", eta = " << genLeg1->eta() << ", phi = " << genLeg1->phi() << std::endl;
      //std::cout << "genLeg2: Pt = " << genLeg2->pt() << ", eta = " << genLeg2->eta() << ", phi = " << genLeg2->phi() << std::endl;
      //std::cout << "genLeg1+2: mass = " << (genLeg1->p4() + genLeg2->p4()).mass() << std::endl;
      if ( !(genLeg1 && genLeg1->energy() > 0. && 
	     genLeg2 && genLeg2->energy() > 0.) ) continue;
      for ( edm::View<reco::Candidate>::const_iterator visDecayProduct1 = visDecayProducts1->begin();
	    visDecayProduct1 != visDecayProducts1->end(); ++visDecayProduct1 ) {
	double dR1matchedTo1 = deltaR(visDecayProduct1->p4(), genLeg1->p4());
	double dR1matchedTo2 = deltaR(visDecayProduct1->p4(), genLeg2->p4());
	for ( edm::View<reco::Candidate>::const_iterator visDecayProduct2 = visDecayProducts2->begin();
	      visDecayProduct2 != visDecayProducts2->end(); ++visDecayProduct2 ) {
	  double dR2matchedTo1 = deltaR(visDecayProduct2->p4(), genLeg1->p4());
	  double dR2matchedTo2 = deltaR(visDecayProduct2->p4(), genLeg2->p4());
	  reco::Candidate::LorentzVector leg1P4;
	  double X1 = 0.;
	  reco::Candidate::LorentzVector leg2P4;
	  double X2 = 0.;
	  bool matched = false;
	  if        ( dR1matchedTo1 < 0.3 && dR2matchedTo1 > 0.5 &&
		      dR2matchedTo2 < 0.3 && dR1matchedTo2 > 0.5 ) {
	    leg1P4 = genLeg1->p4();
	    X1 = visDecayProduct1->energy()/genLeg1->energy();
	    leg2P4 = genLeg2->p4();
	    X2 = visDecayProduct2->energy()/genLeg2->energy();
	    matched = true;
	  } else if ( dR1matchedTo2 < 0.3 && dR2matchedTo2 > 0.5 &&
		      dR2matchedTo1 < 0.3 && dR1matchedTo1 > 0.5 ) {
	    leg1P4 = genLeg2->p4();
	    X1 = visDecayProduct1->energy()/genLeg2->energy();
	    leg2P4 = genLeg1->p4();
	    X2 = visDecayProduct2->energy()/genLeg1->energy();
	    matched = true;
	  } 
	  if ( matched ) {
	    //std::cout << "X1 = " << X1 << ", X2 = " << X2 << std::endl;
	    if ( histogram_leg1Pt && histogram_leg1Eta && histogram_leg1Phi ) {
	      histogram_leg1Pt->Fill(visDecayProduct1->pt(), evtWeight);
	      histogram_leg1Eta->Fill(visDecayProduct1->eta(), evtWeight);
	      histogram_leg1Phi->Fill(visDecayProduct1->phi(), evtWeight);
	    }
	    histogram_leg1X->Fill(X1, evtWeight);
	    if ( histogram_leg2Pt && histogram_leg2Eta && histogram_leg2Phi ) {
	      histogram_leg2Pt->Fill(visDecayProduct2->pt(), evtWeight);
	      histogram_leg2Eta->Fill(visDecayProduct2->eta(), evtWeight);
	      histogram_leg2Phi->Fill(visDecayProduct2->phi(), evtWeight);
	    }
	    histogram_leg2X->Fill(X2, evtWeight);
	  }
	}
      }
    }  
  }
}

void MCEmbeddingValidationAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
//--- compute event weight
  double evtWeight = 1.0;
  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    evtWeight *= (*weight);
  }
  if ( srcGenFilterInfo_.label() != "" ) {
    edm::Handle<GenFilterInfo> genFilterInfo;
    evt.getByLabel(srcGenFilterInfo_, genFilterInfo);
    //std::cout << "genFilterInfo: numEventsTried = " << genFilterInfo->numEventsTried() << ", numEventsPassed = " << genFilterInfo->numEventsPassed() << std::endl;
    if ( genFilterInfo->numEventsTried() > 0 ) {
      double weight = genFilterInfo->filterEfficiency();
      //std::cout << "weight(genFilterInfo) = " << weight << std::endl;
      histogramGenFilterEfficiency_->Fill(weight, evtWeight);
      evtWeight *= weight;
    }
  }

  if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) return;

//--- fill all histograms
  edm::Handle<reco::TrackCollection> tracks;
  evt.getByLabel(srcRecTracks_, tracks);
  int numTracksPtGt5  = 0;
  int numTracksPtGt10 = 0;
  int numTracksPtGt20 = 0;
  int numTracksPtGt30 = 0;
  int numTracksPtGt40 = 0;
  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
    if ( track->pt() >  5. ) ++numTracksPtGt5;
    if ( track->pt() > 10. ) ++numTracksPtGt10;
    if ( track->pt() > 20. ) ++numTracksPtGt20;
    if ( track->pt() > 30. ) ++numTracksPtGt30;
    if ( track->pt() > 40. ) ++numTracksPtGt40;
  }
  histogramNumTracksPtGt5_->Fill(numTracksPtGt5, evtWeight);
  histogramNumTracksPtGt10_->Fill(numTracksPtGt10, evtWeight);
  histogramNumTracksPtGt20_->Fill(numTracksPtGt20, evtWeight);
  histogramNumTracksPtGt30_->Fill(numTracksPtGt30, evtWeight);
  histogramNumTracksPtGt40_->Fill(numTracksPtGt40, evtWeight);
  
 edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcRecPFCandidates_, pfCandidates);
  int numChargedPFCandsPtGt5  = 0;
  int numChargedPFCandsPtGt10 = 0;
  int numChargedPFCandsPtGt20 = 0;
  int numChargedPFCandsPtGt30 = 0;
  int numChargedPFCandsPtGt40 = 0;
  int numNeutralPFCandsPtGt5  = 0;
  int numNeutralPFCandsPtGt10 = 0;
  int numNeutralPFCandsPtGt20 = 0;
  int numNeutralPFCandsPtGt30 = 0;
  int numNeutralPFCandsPtGt40 = 0;
  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    if ( TMath::Abs(pfCandidate->charge()) > 0.5 ) {
      if ( pfCandidate->pt() >  5. ) ++numChargedPFCandsPtGt5;
      if ( pfCandidate->pt() > 10. ) ++numChargedPFCandsPtGt10;
      if ( pfCandidate->pt() > 20. ) ++numChargedPFCandsPtGt20;
      if ( pfCandidate->pt() > 30. ) ++numChargedPFCandsPtGt30;
      if ( pfCandidate->pt() > 40. ) ++numChargedPFCandsPtGt40;
    } else {
      if ( pfCandidate->pt() >  5. ) ++numNeutralPFCandsPtGt5;
      if ( pfCandidate->pt() > 10. ) ++numNeutralPFCandsPtGt10;
      if ( pfCandidate->pt() > 20. ) ++numNeutralPFCandsPtGt20;
      if ( pfCandidate->pt() > 30. ) ++numNeutralPFCandsPtGt30;
      if ( pfCandidate->pt() > 40. ) ++numNeutralPFCandsPtGt40;
    }
  }
  histogramNumChargedPFCandsPtGt5_->Fill(numChargedPFCandsPtGt5, evtWeight);
  histogramNumChargedPFCandsPtGt10_->Fill(numChargedPFCandsPtGt10, evtWeight);
  histogramNumChargedPFCandsPtGt20_->Fill(numChargedPFCandsPtGt20, evtWeight);
  histogramNumChargedPFCandsPtGt30_->Fill(numChargedPFCandsPtGt30, evtWeight);
  histogramNumChargedPFCandsPtGt40_->Fill(numChargedPFCandsPtGt40, evtWeight);
  histogramNumNeutralPFCandsPtGt5_->Fill(numNeutralPFCandsPtGt5, evtWeight);
  histogramNumNeutralPFCandsPtGt10_->Fill(numNeutralPFCandsPtGt10, evtWeight);
  histogramNumNeutralPFCandsPtGt20_->Fill(numNeutralPFCandsPtGt20, evtWeight);
  histogramNumNeutralPFCandsPtGt30_->Fill(numNeutralPFCandsPtGt30, evtWeight);
  histogramNumNeutralPFCandsPtGt40_->Fill(numNeutralPFCandsPtGt40, evtWeight);

  typedef edm::View<reco::Jet> JetView;
  edm::Handle<JetView> jets;
  evt.getByLabel(srcRecJets_, jets);
  int numJetsPtGt20               = 0;
  int numJetsPtGt20AbsEtaLt2_5    = 0;
  int numJetsPtGt20AbsEta2_5to4_5 = 0;
  int numJetsPtGt30               = 0;
  int numJetsPtGt30AbsEtaLt2_5    = 0;
  int numJetsPtGt30AbsEta2_5to4_5 = 0;
  for ( JetView::const_iterator jet = jets->begin();
	jet != jets->end(); ++jet ) {
    double jetPt = jet->pt();
    double absJetEta = TMath::Abs(jet->eta());
    // CV: do not consider any jet reconstructed outside eta range used in H -> tautau analysis
    if ( absJetEta > 4.5 ) continue;
    if ( jetPt > 20. ) {
      ++numJetsPtGt20;
      if      ( absJetEta < 2.5 ) ++numJetsPtGt20AbsEtaLt2_5;
      else if ( absJetEta < 4.5 ) ++numJetsPtGt20AbsEta2_5to4_5;
    }
    if ( jetPt > 30. ) {
      ++numJetsPtGt30;
      if      ( absJetEta < 2.5 ) ++numJetsPtGt30AbsEtaLt2_5;
      else if ( absJetEta < 4.5 ) ++numJetsPtGt30AbsEta2_5to4_5;
    }
  }
  histogramNumJetsPtGt20_->Fill(numJetsPtGt20, evtWeight);
  histogramNumJetsPtGt20AbsEtaLt2_5_->Fill(numJetsPtGt20AbsEtaLt2_5, evtWeight);
  histogramNumJetsPtGt20AbsEta2_5to4_5_->Fill(numJetsPtGt20AbsEta2_5to4_5, evtWeight);
  histogramNumJetsPtGt30_->Fill(numJetsPtGt30, evtWeight);
  histogramNumJetsPtGt30AbsEtaLt2_5_->Fill(numJetsPtGt30AbsEtaLt2_5, evtWeight);
  histogramNumJetsPtGt30AbsEta2_5to4_5_->Fill(numJetsPtGt30AbsEta2_5to4_5, evtWeight);

  edm::Handle<reco::MuonCollection> muons;
  evt.getByLabel(srcRecMuons_, muons);
  int numGlobalMuons     = 0;
  int numStandAloneMuons = 0;
  int numPFMuons         = 0;
  for ( reco::MuonCollection::const_iterator muon = muons->begin();
	muon != muons->end(); ++muon ) {
    if ( muon->isGlobalMuon()     ) ++numGlobalMuons;
    if ( muon->isStandAloneMuon() ) ++numStandAloneMuons;
    if ( muon->isPFMuon()         ) ++numPFMuons;
  }
  histogramNumGlobalMuons_->Fill(numGlobalMuons, evtWeight);
  histogramNumStandAloneMuons_->Fill(numStandAloneMuons, evtWeight);
  histogramNumPFMuons_->Fill(numPFMuons, evtWeight);

  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcRecVertex_, vertices);
  for ( reco::VertexCollection::const_iterator vertex = vertices->begin();
	vertex != vertices->end(); ++vertex ) {
    histogramRecVertexX_->Fill(vertex->position().x(), evtWeight);
    histogramRecVertexY_->Fill(vertex->position().y(), evtWeight);
    histogramRecVertexZ_->Fill(vertex->position().z(), evtWeight);
  }

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> genDiTaus;
  evt.getByLabel(srcGenDiTaus_, genDiTaus);
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    histogramGenDiTauPt_->Fill(genDiTau->pt(), evtWeight);
    histogramGenDiTauEta_->Fill(genDiTau->eta(), evtWeight);
    histogramGenDiTauPhi_->Fill(genDiTau->phi(), evtWeight);
    histogramGenDiTauMass_->Fill(genDiTau->mass(), evtWeight);
  }

  std::vector<reco::CandidateBaseRef> replacedMuons = getSelMuons(evt, srcReplacedMuons_);
  // CV: replacedMuons collection is sorted by decreasing Pt
  bool passesCutsBeforeRotation = false;
  if ( replacedMuons.size() >= 1 && replacedMuons[0]->pt() > replacedMuonPtThresholdHigh_ &&
       replacedMuons.size() >= 2 && replacedMuons[1]->pt() > replacedMuonPtThresholdLow_  ) passesCutsBeforeRotation = true;
  bool passesCutsAfterRotation = false;
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
    if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) continue;
    const reco::Candidate* genTau1 = genDiTau_composite->daughter(0);
    if ( genTau1 ) std::cout << "genTau1: Pt = " << genTau1->pt() << ", phi = " << genTau1->phi() << " (Px = " << genTau1->px() << ", Py = " << genTau1->py() << ")" << std::endl;
    const reco::Candidate* genTau2 = genDiTau_composite->daughter(1);
    if ( genTau2 ) std::cout << "genTau2: Pt = " << genTau2->pt() << ", phi = " << genTau2->phi() << " (Px = " << genTau2->px() << ", Py = " << genTau2->py() << ")" << std::endl;
    if ( !(genTau1 && genTau2) ) continue;
    if ( (genTau1->pt() > replacedMuonPtThresholdHigh_ && genTau2->pt() > replacedMuonPtThresholdLow_ ) ||
	 (genTau1->pt() > replacedMuonPtThresholdLow_  && genTau2->pt() > replacedMuonPtThresholdHigh_) ) {
      passesCutsAfterRotation = true;
      break;
    }
  }
  histogramRotationAngleMatrix_->Fill(passesCutsBeforeRotation, passesCutsAfterRotation, evtWeight);

  fillVisPtEtaPhiMassDistributions(evt, srcGenLeg1_, srcGenLeg2_, histogramGenVisDiTauPt_, histogramGenVisDiTauEta_, histogramGenVisDiTauPhi_, histogramGenVisDiTauMass_, evtWeight);
  fillVisPtEtaPhiMassDistributions(evt, srcRecLeg1_, srcRecLeg2_, histogramRecVisDiTauPt_, histogramRecVisDiTauEta_, histogramRecVisDiTauPhi_, histogramRecVisDiTauMass_, evtWeight);

  fillX1andX2Distributions(evt, srcGenDiTaus_, srcGenLeg1_, srcGenLeg2_, 
			   histogramGenLeg1Pt_, histogramGenLeg1Eta_, histogramGenLeg1Phi_, histogramGenLeg1X_, 
			   histogramGenLeg2Pt_, histogramGenLeg2Eta_, histogramGenLeg2Phi_, histogramGenLeg2X_, evtWeight);
  fillX1andX2Distributions(evt, srcGenDiTaus_, srcRecLeg1_, srcRecLeg2_, 0, 0, 0, histogramRecLeg1X_, 0, 0, 0, histogramRecLeg2X_, evtWeight);
  
  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(srcGenParticles_, genParticles);     
  reco::Candidate::LorentzVector sumGenParticleP4;
  reco::Candidate::LorentzVector sumGenParticleP4_charged;
  for ( reco::GenParticleCollection::const_iterator genParticle = genParticles->begin();
	genParticle != genParticles->end(); ++genParticle ) {
    if ( genParticle->status() == 1 ) {
      int absPdgId = TMath::Abs(genParticle->pdgId());    
      if ( absPdgId == 12 || absPdgId == 14 || absPdgId == 16 ) continue;
      sumGenParticleP4 += genParticle->p4();
      if ( TMath::Abs(genParticle->charge()) > 0.5 ) sumGenParticleP4_charged += genParticle->p4();
    }
  }
  histogramSumGenParticlePt_->Fill(sumGenParticleP4.pt(), evtWeight);

  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> genMETs;
  evt.getByLabel(srcGenMEt_, genMETs);
  const reco::Candidate::LorentzVector& genMEtP4 = genMETs->front().p4();
  std::cout << "genMEt: Pt = " << genMEtP4.pt() << ", phi = " << genMEtP4.phi() << " (Px = " << genMEtP4.px() << ", Py = " << genMEtP4.py() << ")" << std::endl;
  histogramGenMEt_->Fill(genMEtP4.pt(), evtWeight);

  double sumTracksPx = 0.;
  double sumTracksPy = 0.;
  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
    sumTracksPx += track->px();
    sumTracksPy += track->py();
  }
  histogramRecTrackMEt_->Fill(TMath::Sqrt(sumTracksPx*sumTracksPx + sumTracksPy*sumTracksPy), evtWeight);

  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel(srcCaloTowers_, caloTowers);
  reco::Candidate::LorentzVector sumCaloTowerP4_ecal;
  double sumEtCaloTowersECAL = 0.;
  reco::Candidate::LorentzVector sumCaloTowerP4_hcal;
  double sumEtCaloTowersHCAL = 0.;
  reco::Candidate::LorentzVector sumCaloTowerP4_hf;
  double sumEtCaloTowersHF   = 0.;
  reco::Candidate::LorentzVector sumCaloTowerP4_ho;
  double sumEtCaloTowersHO   = 0.;
  for ( CaloTowerCollection::const_iterator caloTower = caloTowers->begin();
	caloTower != caloTowers->end(); ++caloTower ) {
    if ( caloTower->energy() != 0. ) {
      double emFrac_ecal = caloTower->emEnergy()/caloTower->energy();
      double emFrac_hcal = (caloTower->energyInHB() + caloTower->energyInHE())/caloTower->energy();
      double emFrac_hf   = caloTower->energyInHF()/caloTower->energy();
      double emFrac_ho   = caloTower->energyInHO()/caloTower->energy();
      sumCaloTowerP4_ecal += (emFrac_ecal*caloTower->p4());
      sumEtCaloTowersECAL += (emFrac_ecal*caloTower->et());
      sumCaloTowerP4_hcal += (emFrac_hcal*caloTower->p4());
      sumEtCaloTowersHCAL += (emFrac_hcal*caloTower->et());
      sumCaloTowerP4_hf   += (emFrac_hf*caloTower->p4());
      sumEtCaloTowersHF   += (emFrac_hf*caloTower->et());
      sumCaloTowerP4_ho   += (emFrac_ho*caloTower->p4());      
      sumEtCaloTowersHO   += (emFrac_ho*caloTower->et());  
    }
  }
  histogramRecCaloMEtECAL_->Fill(sumCaloTowerP4_ecal.pt(), evtWeight);
  histogramRecCaloSumEtECAL_->Fill(sumEtCaloTowersECAL, evtWeight);
  histogramRecCaloMEtHCAL_->Fill(sumCaloTowerP4_hcal.pt(), evtWeight);
  histogramRecCaloSumEtHCAL_->Fill(sumEtCaloTowersHCAL, evtWeight);
  histogramRecCaloMEtHF_->Fill(sumCaloTowerP4_hf.pt(), evtWeight);
  histogramRecCaloSumEtHF_->Fill(sumEtCaloTowersHF, evtWeight);
  histogramRecCaloMEtHO_->Fill(sumCaloTowerP4_ho.pt(), evtWeight);
  histogramRecCaloSumEtHO_->Fill(sumEtCaloTowersHO, evtWeight);
  
  edm::Handle<l1extra::L1EtMissParticleCollection> l1METs;
  evt.getByLabel(srcL1ETM_, l1METs);
  const reco::Candidate::LorentzVector& l1MEtP4 = l1METs->front().p4();
  std::cout << "L1MEt: Pt = " << l1MEtP4.pt() << ", phi = " << l1MEtP4.phi() << " (Et = " << l1METs->front().etMiss() << ", Px = " << l1MEtP4.px() << ", Py = " << l1MEtP4.py() << ")" << std::endl;
  if ( l1MEtP4.pt() > 75. ) std::cout << "--> CHECK !!" << std::endl;
  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> recCaloMETs;
  evt.getByLabel(srcRecCaloMEt_, recCaloMETs);
  const reco::Candidate::LorentzVector& recCaloMEtP4 = recCaloMETs->front().p4();
  std::cout << "recCaloMEt: Pt = " << recCaloMEtP4.pt() << ", phi = " << recCaloMEtP4.phi() << " (Px = " << recCaloMEtP4.px() << ", Py = " << recCaloMEtP4.py() << ")" << std::endl;
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
    if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) continue;
    const reco::Candidate* genDaughter1 = genDiTau_composite->daughter(0);
    const reco::GenParticle* genTau1 = ( genDaughter1->hasMasterClone() ) ?
      dynamic_cast<const reco::GenParticle*>(&(*genDaughter1->masterClone())) : dynamic_cast<const reco::GenParticle*>(genDaughter1);
    const reco::Candidate* genDaughter2 = genDiTau_composite->daughter(1);
    const reco::GenParticle* genTau2 = ( genDaughter2->hasMasterClone() ) ?
      dynamic_cast<const reco::GenParticle*>(&(*genDaughter2->masterClone())) : dynamic_cast<const reco::GenParticle*>(genDaughter2);
    if ( !(genTau1 && genTau2) ) continue;
    std::string genTauDecayMode1 = getGenTauDecayMode(genTau1);
    std::string genTauDecayMode2 = getGenTauDecayMode(genTau2);
    std::string genTauDecayMode_ref;
    if      ( genTauDecayMode1 == "electron" || genTauDecayMode1 == "muon" ) genTauDecayMode_ref = genTauDecayMode2;
    else if ( genTauDecayMode2 == "electron" || genTauDecayMode2 == "muon" ) genTauDecayMode_ref = genTauDecayMode1;
    for ( std::vector<plotEntryTypeL1ETM*>::iterator l1ETMplotEntry = l1ETMplotEntries_.begin();
	  l1ETMplotEntry != l1ETMplotEntries_.end(); ++l1ETMplotEntry ) {
      (*l1ETMplotEntry)->fillHistograms(genTauDecayMode_ref, l1MEtP4, genMEtP4, recCaloMEtP4, genDiTau->p4(), evtWeight);
    }
  }

  fillHistograms(electronDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(electronEfficiencies_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(electronL1TriggerEfficiencies_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(muonDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(muonEfficiencies_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(muonL1TriggerEfficiencies_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(tauDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(tauDistributionsExtra_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(tauEfficiencies_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(l1ElectronDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(l1MuonDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(l1TauDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(l1CentralJetDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(l1ForwardJetDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(metDistributions_, numJetsPtGt30, evt, evtWeight);
  fillHistograms(metL1TriggerEfficiencies_, numJetsPtGt30, evt, evtWeight);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCEmbeddingValidationAnalyzer);
