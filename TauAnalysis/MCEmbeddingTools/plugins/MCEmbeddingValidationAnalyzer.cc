#include "TauAnalysis/MCEmbeddingTools/plugins/MCEmbeddingValidationAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

MCEmbeddingValidationAnalyzer::MCEmbeddingValidationAnalyzer(const edm::ParameterSet& cfg)
  : srcWeights_(cfg.getParameter<vInputTag>("srcWeights")),
    dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory"))
{
//--- setup electron Pt, eta and phi distributions;
//    electron id & isolation and trigger efficiencies
  setupLeptonDistribution(cfg, "electronDistributions", electronDistributions_);
  setupLeptonEfficiency(cfg, "electronEfficiencies", electronEfficiencies_);
  setupLeptonL1TriggerEfficiency(cfg, "electronL1TriggerEfficiencies", electronL1TriggerEfficiencies_);

//--- setup muon Pt, eta and phi distributions;
//    muon id & isolation and trigger efficiencies
  setupLeptonDistribution(cfg, "muonDistributions", muonDistributions_);
  setupLeptonEfficiency(cfg, "muonEfficiencies", muonEfficiencies_);
  setupLeptonL1TriggerEfficiency(cfg, "muonL1TriggerEfficiencies", muonL1TriggerEfficiencies_);
  
//--- setup tau Pt, eta and phi distributions;
//    tau id efficiency
  setupLeptonDistribution(cfg, "tauDistributions", tauDistributions_);
  setupLeptonEfficiency(cfg, "tauEfficiencies", tauEfficiencies_);

//--- setup MET Pt and phi distributions;
//    efficiency of L1 (Calo)MET trigger requirement
  setupMEtDistribution(cfg, "metDistributions", metDistributions_);
  setupMEtL1TriggerEfficiency(cfg, "metL1TriggerEfficiencies", metL1TriggerEfficiencies_);
}

MCEmbeddingValidationAnalyzer::~MCEmbeddingValidationAnalyzer()
{
  cleanCollection(electronDistributions_);
  cleanCollection(electronEfficiencies_);
  cleanCollection(electronL1TriggerEfficiencies_);
  cleanCollection(muonDistributions_);
  cleanCollection(muonEfficiencies_);
  cleanCollection(muonL1TriggerEfficiencies_);
  cleanCollection(tauDistributions_);
  cleanCollection(tauEfficiencies_);
  cleanCollection(metDistributions_);
  cleanCollection(metL1TriggerEfficiencies_);
}

template <typename T>
void MCEmbeddingValidationAnalyzer::setupLeptonDistribution(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonDistributionT<T>*>& leptonDistributions)
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
      leptonDistributionT<T>* leptonDistribution = new leptonDistributionT<T>(srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      leptonDistributions.push_back(leptonDistribution);
    }
  }
}

template <typename T>
void MCEmbeddingValidationAnalyzer::setupLeptonEfficiency(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonEfficiencyT<T>*>& leptonEfficiencies)
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
      leptonEfficiencyT<T>* leptonEfficiency = new leptonEfficiencyT<T>(srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      leptonEfficiencies.push_back(leptonEfficiency);
    }
  }
}

template <typename T1, typename T2>
void MCEmbeddingValidationAnalyzer::setupLeptonL1TriggerEfficiency(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<leptonL1TriggerEfficiencyT1T2<T1,T2>*>& leptonL1TriggerEfficiencies)
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
      leptonL1TriggerEfficiencyT1T2<T1,T2>* leptonL1TriggerEfficiency = new leptonL1TriggerEfficiencyT1T2<T1,T2>(srcRef, cutRef, srcL1, cutL1, dRmatch, dqmDirectory);
      leptonL1TriggerEfficiencies.push_back(leptonL1TriggerEfficiency);
    }
  }
}

void MCEmbeddingValidationAnalyzer::setupMEtDistribution(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<metDistributionType*>& metDistributions)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgMEtDistributions = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgMEtDistribution = cfgMEtDistributions.begin();
	  cfgMEtDistribution != cfgMEtDistributions.end(); ++cfgMEtDistribution ) {
      edm::InputTag srcGen = cfgMEtDistribution->getParameter<edm::InputTag>("srcGen");
      edm::InputTag srcRec = cfgMEtDistribution->getParameter<edm::InputTag>("srcRec");
      edm::InputTag srcGenZs = cfgMEtDistribution->getParameter<edm::InputTag>("srcGenZs");
      std::string dqmDirectory = dqmDirectory_full(cfgMEtDistribution->getParameter<std::string>("dqmDirectory"));
      metDistributionType* metDistribution = new metDistributionType(srcGen, srcRec, srcGenZs, dqmDirectory);
      metDistributions.push_back(metDistribution);
    }
  }
}
    
void MCEmbeddingValidationAnalyzer::setupMEtL1TriggerEfficiency(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<metL1TriggerEfficiencyType*>& metL1TriggerEfficiencies)
{
  if ( cfg.exists(keyword) ) {
    edm::VParameterSet cfgMEtL1TriggerEfficiencies = cfg.getParameter<edm::VParameterSet>(keyword);
    for ( edm::VParameterSet::const_iterator cfgMEtL1TriggerEfficiency = cfgMEtL1TriggerEfficiencies.begin();
	  cfgMEtL1TriggerEfficiency != cfgMEtL1TriggerEfficiencies.end(); ++cfgMEtL1TriggerEfficiency ) {
      edm::InputTag srcRef = cfgMEtL1TriggerEfficiency->getParameter<edm::InputTag>("srcRef");
      edm::InputTag srcL1 = cfgMEtL1TriggerEfficiency->getParameter<edm::InputTag>("srcL1");
      double cutL1Pt = cfgMEtL1TriggerEfficiency->getParameter<double>("cutL1Pt");
      std::string dqmDirectory = dqmDirectory_full(cfgMEtL1TriggerEfficiency->getParameter<std::string>("dqmDirectory"));
      metL1TriggerEfficiencyType* metL1TriggerEfficiency = new metL1TriggerEfficiencyType(srcRef, srcL1, cutL1Pt, dqmDirectory);
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
  bookHistograms(electronDistributions_, dqmStore);
  bookHistograms(electronEfficiencies_, dqmStore);
  bookHistograms(electronL1TriggerEfficiencies_, dqmStore);
  bookHistograms(muonDistributions_, dqmStore);
  bookHistograms(muonEfficiencies_, dqmStore);
  bookHistograms(muonL1TriggerEfficiencies_, dqmStore);
  bookHistograms(tauDistributions_, dqmStore);
  bookHistograms(tauEfficiencies_, dqmStore);
  bookHistograms(metDistributions_, dqmStore);
  bookHistograms(metL1TriggerEfficiencies_, dqmStore);
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

  if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) return;

//--- fill all histograms
  fillHistograms(electronDistributions_, evt, evtWeight);
  fillHistograms(electronEfficiencies_, evt, evtWeight);
  fillHistograms(electronL1TriggerEfficiencies_, evt, evtWeight);
  fillHistograms(muonDistributions_, evt, evtWeight);
  fillHistograms(muonEfficiencies_, evt, evtWeight);
  fillHistograms(muonL1TriggerEfficiencies_, evt, evtWeight);
  fillHistograms(tauDistributions_, evt, evtWeight);
  fillHistograms(tauEfficiencies_, evt, evtWeight);
  fillHistograms(metDistributions_, evt, evtWeight);
  fillHistograms(metL1TriggerEfficiencies_, evt, evtWeight);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCEmbeddingValidationAnalyzer);
