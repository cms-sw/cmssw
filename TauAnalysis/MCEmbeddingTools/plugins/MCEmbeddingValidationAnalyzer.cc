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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Math/interface/normalizedPhi.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <Math/VectorUtil.h>
#include <TMath.h>

#include <sstream>

// Copied from TauAnalysis/CandidateTools/interface/svFitAuxFunctions.cc. Note that we cannot use
// this package directly, since it is not part of CMSSW, but we are.
namespace SVfit_namespace
{
  inline double square(double x)
  {
    return x*x;
  }

  reco::Candidate::Vector normalize(const reco::Candidate::Vector& p)
  {
    double p_x = p.x();
    double p_y = p.y();
    double p_z = p.z();
    double mag2 = square(p_x) + square(p_y) + square(p_z);
    if ( mag2 <= 0. ) return p;
    double mag = TMath::Sqrt(mag2);
    return reco::Candidate::Vector(p_x/mag, p_y/mag, p_z/mag);
  }

  double compScalarProduct(const reco::Candidate::Vector& p1, const reco::Candidate::Vector& p2)
  {
    return (p1.x()*p2.x() + p1.y()*p2.y() + p1.z()*p2.z());
  }
  
  reco::Candidate::Vector compCrossProduct(const reco::Candidate::Vector& p1, const reco::Candidate::Vector& p2)
  {
    double p3_x = p1.y()*p2.z() - p1.z()*p2.y();
    double p3_y = p1.z()*p2.x() - p1.x()*p2.z();
    double p3_z = p1.x()*p2.y() - p1.y()*p2.x();
    return reco::Candidate::Vector(p3_x, p3_y, p3_z);
  }

  double phiLabFromLabMomenta(const reco::Candidate::LorentzVector& motherP4, const reco::Candidate::LorentzVector& visP4)
  {
    reco::Candidate::Vector u_z = normalize(reco::Candidate::Vector(visP4.px(), visP4.py(), visP4.pz()));
    reco::Candidate::Vector u_y = normalize(compCrossProduct(reco::Candidate::Vector(0., 0., 1.), u_z));
    reco::Candidate::Vector u_x = compCrossProduct(u_y, u_z);
    
    reco::Candidate::Vector p3Mother_unit = normalize(reco::Candidate::Vector(motherP4.px(), motherP4.py(), motherP4.pz()));
    
    double phi_lab = TMath::ATan2(compScalarProduct(p3Mother_unit, u_y), compScalarProduct(p3Mother_unit, u_x));
    return phi_lab;
  }
}

/*EGammaMvaEleEstimator* MCEmbeddingValidationAnalyzer::electronDistributionExtra::fMVA_ = 0;
bool MCEmbeddingValidationAnalyzer::electronDistributionExtra::fMVA_isInitialized_ = false;*/

MCEmbeddingValidationAnalyzer::MCEmbeddingValidationAnalyzer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),    
    srcReplacedMuons_(cfg.getParameter<edm::InputTag>("srcReplacedMuons")),
    srcRecMuons_(cfg.getParameter<edm::InputTag>("srcRecMuons")),
    srcRecTracks_(cfg.getParameter<edm::InputTag>("srcRecTracks")),
    srcCaloTowers_(cfg.getParameter<edm::InputTag>("srcCaloTowers")),
    srcRecPFCandidates_(cfg.getParameter<edm::InputTag>("srcRecPFCandidates")),
    srcRecJets_(cfg.getParameter<edm::InputTag>("srcRecJets")),
    srcTheRecVertex_(cfg.getParameter<edm::InputTag>("srcTheRecVertex")),
    srcRecVertices_(cfg.getParameter<edm::InputTag>("srcRecVertices")),
    srcRecVerticesWithBS_(cfg.getParameter<edm::InputTag>("srcRecVerticesWithBS")),
    srcBeamSpot_(cfg.getParameter<edm::InputTag>("srcBeamSpot")),
    srcGenDiTaus_(cfg.getParameter<edm::InputTag>("srcGenDiTaus")),
    dRminSeparation_(cfg.getParameter<double>("dRminSeparation")),
    srcGenLeg1_(cfg.getParameter<edm::InputTag>("srcGenLeg1")),
    srcRecLeg1_(cfg.getParameter<edm::InputTag>("srcRecLeg1")),
    srcGenLeg2_(cfg.getParameter<edm::InputTag>("srcGenLeg2")),
    srcRecLeg2_(cfg.getParameter<edm::InputTag>("srcRecLeg2")),
    srcGenParticles_(cfg.getParameter<edm::InputTag>("srcGenParticles")),
    srcL1ETM_(cfg.getParameter<edm::InputTag>("srcL1ETM")),
    srcGenCaloMEt_(cfg.getParameter<edm::InputTag>("srcGenCaloMEt")),
    srcGenPFMEt_(cfg.getParameter<edm::InputTag>("srcGenPFMEt")),
    srcRecCaloMEt_(cfg.getParameter<edm::InputTag>("srcRecCaloMEt")),
    srcRecPFMEt_(cfg.getParameter<edm::InputTag>("srcRecPFMEt")),
    srcMuonsBeforeRad_(cfg.getParameter<edm::InputTag>("srcMuonsBeforeRad")),
    srcMuonsAfterRad_(cfg.getParameter<edm::InputTag>("srcMuonsAfterRad")),
    srcMuonRadCorrWeight_(cfg.getParameter<edm::InputTag>("srcMuonRadCorrWeight")),
    srcMuonRadCorrWeightUp_(cfg.getParameter<edm::InputTag>("srcMuonRadCorrWeightUp")),
    srcMuonRadCorrWeightDown_(cfg.getParameter<edm::InputTag>("srcMuonRadCorrWeightDown")),
    srcOtherWeights_(cfg.getParameter<vInputTag>("srcOtherWeights")),
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
    setupElectronDistributionExtra(jetBin->first, jetBin->second, cfg, "electronDistributions", electronDistributionsExtra_);
    setupLeptonEfficiency(jetBin->first, jetBin->second, cfg, "electronEfficiencies", electronEfficiencies_);
    setupLeptonEfficiency(jetBin->first, jetBin->second, cfg, "gsfElectronEfficiencies", gsfElectronEfficiencies_);
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

  ebRHToken_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
  eeRHToken_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

MCEmbeddingValidationAnalyzer::~MCEmbeddingValidationAnalyzer()
{
  for ( std::vector<plotEntryTypeEvtWeight*>::iterator it = evtWeightPlotEntries_.begin();
	it != evtWeightPlotEntries_.end(); ++it ) {
    delete (*it);
  }

  for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator it = muonRadCorrUncertaintyPlotEntries_beforeRad_.begin();
	it != muonRadCorrUncertaintyPlotEntries_beforeRad_.end(); ++it ) {
    delete (*it);
  }
  for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator it = muonRadCorrUncertaintyPlotEntries_afterRad_.begin();
	it != muonRadCorrUncertaintyPlotEntries_afterRad_.end(); ++it ) {
    delete (*it);
  }
  for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator it = muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_.begin();
	it != muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_.end(); ++it ) {
    delete (*it);
  }

  for ( std::vector<plotEntryTypeL1ETM*>::iterator it = l1ETMplotEntries_.begin();
	it != l1ETMplotEntries_.end(); ++it ) {
    delete (*it);
  }

  cleanCollection(electronDistributions_);
  cleanCollection(electronDistributionsExtra_);
  cleanCollection(electronEfficiencies_);
  cleanCollection(gsfElectronEfficiencies_);
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

void MCEmbeddingValidationAnalyzer::setupElectronDistributionExtra(int minJets, int maxJets,
								   const edm::ParameterSet& cfg, const std::string& keyword, std::vector<electronDistributionExtra*>& electronDistributionsExtra)
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
      electronDistributionExtra* electronDistribution = new electronDistributionExtra(minJets, maxJets, srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory, srcTheRecVertex_, ebRHToken_, eeRHToken_);
      electronDistributionsExtra.push_back(electronDistribution);
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
  dqmStore.setCurrentFolder(dqmDirectory_.data());

//--- book all histograms
  histogramEventCounter_                       = dqmStore.book1D("EventCounter",                       "EventCounter",                              1,     -0.5,          1.5);
 
  histogramGenFilterEfficiency_                = dqmStore.book1D("genFilterEfficiency",                "genFilterEfficiency",                     102,     -0.01,         1.01);

  histogramRotationAngleMatrix_                = dqmStore.book2D("rfRotationAngleMatrix",              "rfRotationAngleMatrix",                     2,     -0.5,          1.5, 2, -0.5, 1.5);
  histogramRotationLegPlusDeltaR_              = dqmStore.book1D("rfRotationLegPlusDeltaR",            "rfRotationLegPlusDeltaR",                 101,     -0.05,        10.05);
  histogramRotationLegMinusDeltaR_             = dqmStore.book1D("rfRotationLegMinusDeltaR",           "rfRotationLegMinusDeltaR",                101,     -0.05,        10.05);
  histogramPhiRotLegPlus_                      = dqmStore.book1D("rfPhiRotLegPlus",                    "rfPhiRotLegPlus",                          72, -TMath::Pi(), +TMath::Pi());
  histogramPhiRotLegMinus_                     = dqmStore.book1D("rfPhiRotLegMinus",                   "rfPhiRotLegMinus",                         72, -TMath::Pi(), +TMath::Pi());

  histogramNumTracksPtGt5_                     = dqmStore.book1D("numTracksPtGt5",                     "numTracksPtGt5",                           50,     -0.5,         49.5);
  histogramNumTracksPtGt10_                    = dqmStore.book1D("numTracksPtGt10",                    "numTracksPtGt10",                          50,     -0.5,         49.5);
  histogramNumTracksPtGt20_                    = dqmStore.book1D("numTracksPtGt20",                    "numTracksPtGt20",                          50,     -0.5,         49.5);
  histogramNumTracksPtGt30_                    = dqmStore.book1D("numTracksPtGt30",                    "numTracksPtGt30",                          50,     -0.5,         49.5);
  histogramNumTracksPtGt40_                    = dqmStore.book1D("numTracksPtGt40",                    "numTracksPtGt40",                          50,     -0.5,         49.5);
      
  histogramNumGlobalMuons_                     = dqmStore.book1D("numGlobalMuons",                     "numGlobalMuons",                           20,     -0.5,         19.5);
  histogramNumStandAloneMuons_                 = dqmStore.book1D("numStandAloneMuons",                 "numStandAloneMuons",                       20,     -0.5,         19.5);
  histogramNumPFMuons_                         = dqmStore.book1D("numPFMuons",                         "numPFMuons",                               20,     -0.5,         19.5);

  histogramNumChargedPFCandsPtGt5_             = dqmStore.book1D("numChargedPFCandsPtGt5",             "numChargedPFCandsPtGt5",                   50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt10_            = dqmStore.book1D("numChargedPFCandsPtGt10",            "numChargedPFCandsPtGt10",                  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt20_            = dqmStore.book1D("numChargedPFCandsPtGt20",            "numChargedPFCandsPtGt20",                  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt30_            = dqmStore.book1D("numChargedPFCandsPtGt30",            "numChargedPFCandsPtGt30",                  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt40_            = dqmStore.book1D("numChargedPFCandsPtGt40",            "numChargedPFCandsPtGt40",                  50,     -0.5,         49.5);

  histogramNumNeutralPFCandsPtGt5_             = dqmStore.book1D("numNeutralPFCandsPtGt5",             "numNeutralPFCandsPtGt5",                   50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt10_            = dqmStore.book1D("numNeutralPFCandsPtGt10",            "numNeutralPFCandsPtGt10",                  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt20_            = dqmStore.book1D("numNeutralPFCandsPtGt20",            "numNeutralPFCandsPtGt20",                  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt30_            = dqmStore.book1D("numNeutralPFCandsPtGt30",            "numNeutralPFCandsPtGt30",                  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt40_            = dqmStore.book1D("numNeutralPFCandsPtGt40",            "numNeutralPFCandsPtGt40",                  50,     -0.5,         49.5);
    
  histogramRawJetPt_                           = dqmStore.book1D("rawJetPt",                           "rawJetPt",                                250,      0.,         250.);
  histogramRawJetPtAbsEtaLt2_5_                = dqmStore.book1D("rawJetPtAbsEtaLt2_5",                "rawJetPtAbsEtaLt2_5",                     250,      0.,         250.);  
  histogramRawJetPtAbsEta2_5to4_5_             = dqmStore.book1D("rawJetPtAbsEta2_5to4_5",             "rawJetPtAbsEta2_5to4_5",                  250,      0.,         250.);
  histogramRawJetEtaPtGt20_                    = dqmStore.book1D("rawJetEtaPtGt20",                    "rawJetEtaPtGt20",                         198,     -9.9,         +9.9);
  histogramRawJetEtaPtGt30_                    = dqmStore.book1D("rawJetEtaPtGt30",                    "rawJetEtaPtGt30",                         198,     -9.9,         +9.9);
  histogramNumJetsRawPtGt20_                   = dqmStore.book1D("numJetsRawPtGt20",                   "numJetsRawPtGt20",                         50,     -0.5,         49.5);
  histogramNumJetsRawPtGt20AbsEtaLt2_5_        = dqmStore.book1D("numJetsRawPtGt20AbsEtaLt2_5",        "numJetsRawPtGt20AbsEtaLt2_5",              50,     -0.5,         49.5);
  histogramNumJetsRawPtGt20AbsEta2_5to4_5_     = dqmStore.book1D("numJetsRawPtGt20AbsEta2_5to4_5",     "numJetsRawPtGt20AbsEta2_5to4_5",           50,     -0.5,         49.5);
  histogramNumJetsRawPtGt30_                   = dqmStore.book1D("numJetsRawPtGt30",                   "numJetsRawPtGt30",                         50,     -0.5,         49.5);
  histogramNumJetsRawPtGt30AbsEtaLt2_5_        = dqmStore.book1D("numJetsRawPtGt30AbsEtaLt2_5",        "numJetsRawPtGt30AbsEtaLt2_5",              50,     -0.5,         49.5);
  histogramNumJetsRawPtGt30AbsEta2_5to4_5_     = dqmStore.book1D("numJetsRawPtGt30AbsEta2_5to4_5",     "numJetsRawPtGt30AbsEta2_5to4_5",           50,     -0.5,         49.5);
  histogramCorrJetPt_                          = dqmStore.book1D("corrJetPt",                          "corrJetPt",                               250,      0.,         250.);
  histogramCorrJetPtAbsEtaLt2_5_               = dqmStore.book1D("corrJetPtAbsEtaLt2_5",               "corrJetPtAbsEtaLt2_5",                    250,      0.,         250.);  
  histogramCorrJetPtAbsEta2_5to4_5_            = dqmStore.book1D("corrJetPtAbsEta2_5to4_5",            "corrJetPtAbsEta2_5to4_5",                 250,      0.,         250.);
  histogramCorrJetEtaPtGt20_                   = dqmStore.book1D("corrJetEtaPtGt20",                   "corrJetEtaPtGt20",                        198,     -9.9,         +9.9);
  histogramCorrJetEtaPtGt30_                   = dqmStore.book1D("corrJetEtaPtGt30",                   "corrJetEtaPtGt30",                        198,     -9.9,         +9.9);
  histogramNumJetsCorrPtGt20_                  = dqmStore.book1D("numJetsCorrPtGt20",                  "numJetsCorrPtGt20",                        20,     -0.5,         19.5);
  histogramNumJetsCorrPtGt20AbsEtaLt2_5_       = dqmStore.book1D("numJetsCorrPtGt20AbsEtaLt2_5",       "numJetsCorrPtGt20AbsEtaLt2_5",             20,     -0.5,         19.5);
  histogramNumJetsCorrPtGt20AbsEta2_5to4_5_    = dqmStore.book1D("numJetsCorrPtGt20AbsEta2_5to4_5",    "numJetsCorrPtGt20AbsEta2_5to4_5",          20,     -0.5,         19.5);  
  histogramNumJetsCorrPtGt30_                  = dqmStore.book1D("numJetsCorrPtGt30",                  "numJetsCorrPtGt30",                        20,     -0.5,         19.5);
  histogramNumJetsCorrPtGt30AbsEtaLt2_5_       = dqmStore.book1D("numJetsCorrPtGt30AbsEtaLt2_5",       "numJetsCorrPtGt30AbsEtaLt2_5",             20,     -0.5,         19.5);
  histogramNumJetsCorrPtGt30AbsEta2_5to4_5_    = dqmStore.book1D("numJetsCorrPtGt30AbsEta2_5to4_5",    "numJetsCorrPtGt30AbsEta2_5to4_5",          20,     -0.5,         19.5);
    
  histogramTheRecVertexX_                      = dqmStore.book1D("theRecVertexX",                      "theRecVertexX",                          2000,     -1.,          +1.);
  histogramTheRecVertexY_                      = dqmStore.book1D("theRecVertexY",                      "theRecVertexY",                          2000,     -1.,          +1.);
  histogramTheRecVertexZ_                      = dqmStore.book1D("theRecVertexZ",                      "theRecVertexZ",                           500,    -25.,         +25.);
  histogramRecVertexX_                         = dqmStore.book1D("recVertexX",                         "recVertexX",                             2000,     -1.,          +1.);
  histogramRecVertexY_                         = dqmStore.book1D("recVertexY",                         "recVertexY",                             2000,     -1.,          +1.);
  histogramRecVertexZ_                         = dqmStore.book1D("recVertexZ",                         "recVertexZ",                              500,    -25.,         +25.);
  histogramNumRecVertices_                     = dqmStore.book1D("numRecVertices",                     "numRecVertices",                           50,     -0.5,        +49.5);
  histogramRecVertexWithBSx_                   = dqmStore.book1D("recVertexWithBSx",                   "recVertexWithBSx",                       2000,     -1.,          +1.);
  histogramRecVertexWithBSy_                   = dqmStore.book1D("recVertexWithBSy",                   "recVertexWithBSy",                       2000,     -1.,          +1.);
  histogramRecVertexWithBSz_                   = dqmStore.book1D("recVertexWithBSz",                   "recVertexWithBSz",                        500,    -25.,         +25.);
  histogramNumRecVerticesWithBS_               = dqmStore.book1D("numRecVerticesWithBS",               "numRecVerticesWithBS",                     50,     -0.5,        +49.5);

  histogramBeamSpotX_                          = dqmStore.book1D("beamSpotX",                          "beamSpotX",                              2000,     -1.,          +1.);
  histogramBeamSpotY_                          = dqmStore.book1D("beamSpotY",                          "beamSpotY",                              2000,     -1.,          +1.);
  
  histogramGenDiTauPt_                         = dqmStore.book1D("genDiTauPt",                         "genDiTauPt",                              250,      0.,         250.);
  histogramGenDiTauEta_                        = dqmStore.book1D("genDiTauEta",                        "genDiTauEta",                             198,     -9.9,         +9.9);
  histogramGenDiTauPhi_                        = dqmStore.book1D("genDiTauPhi",                        "genDiTauPhi",                              72, -TMath::Pi(), +TMath::Pi());
  histogramGenDiTauMass_                       = dqmStore.book1D("genDiTauMass",                       "genDiTauMass",                            500,      0.,         500.);
  histogramGenDeltaPhiLeg1Leg2_                = dqmStore.book1D("genDeltaPhiLeg1Leg2",                "genDeltaPhiLeg1Leg2",                     180,      0.,      +TMath::Pi());
  histogramGenDiTauDecayAngle_                 = dqmStore.book1D("genDiTauDecayAngle",                 "genDiTauDecayAngle",                      180,      0.,      +TMath::Pi());

  histogramGenVisDiTauPt_                      = dqmStore.book1D("genVisDiTauPt",                      "genVisDiTauPt",                           250,      0.,         250.);
  histogramGenVisDiTauEta_                     = dqmStore.book1D("genVisDiTauEta",                     "genVisDiTauEta",                          198,     -9.9,         +9.9);
  histogramGenVisDiTauPhi_                     = dqmStore.book1D("genVisDiTauPhi",                     "genVisDiTauPhi",                           72, -TMath::Pi(), +TMath::Pi());
  histogramGenVisDiTauMass_                    = dqmStore.book1D("genVisDiTauMass",                    "genVisDiTauMass",                         500,      0.,         500.);
  histogramGenVisDeltaPhiLeg1Leg2_             = dqmStore.book1D("genVisDeltaPhiLeg1Leg2",             "genVisDeltaPhiLeg1Leg2",                  180,      0.,      +TMath::Pi());

  histogramRecVisDiTauPt_                      = dqmStore.book1D("recVisDiTauPt",                      "recVisDiTauPt",                           250,      0.,         250.);
  histogramRecVisDiTauEta_                     = dqmStore.book1D("recVisDiTauEta",                     "recVisDiTauEta",                          198,     -9.9,         +9.9);
  histogramRecVisDiTauPhi_                     = dqmStore.book1D("recVisDiTauPhi",                     "recVisDiTauPhi",                           72, -TMath::Pi(), +TMath::Pi());
  histogramRecVisDiTauMass_                    = dqmStore.book1D("recVisDiTauMass",                    "recVisDiTauMass",                         500,      0.,         500.);
  histogramRecVisDeltaPhiLeg1Leg2_             = dqmStore.book1D("recVisDeltaPhiLeg1Leg2",             "recVisDeltaPhiLeg1Leg2",                  180,      0.,      +TMath::Pi());

  histogramGenTau1Pt_                          = dqmStore.book1D("genTau1Pt",                          "genTau1Pt",                               250,      0.,         250.);
  histogramGenTau1Eta_                         = dqmStore.book1D("genTau1Eta",                         "genTau1Eta",                              198,     -9.9,         +9.9);
  histogramGenTau1Phi_                         = dqmStore.book1D("genTau1Phi",                         "genTau1Phi",                               72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg1Pt_                          = dqmStore.book1D("genLeg1Pt",                          "genLeg1Pt",                               250,      0.,         250.);
  histogramGenLeg1Eta_                         = dqmStore.book1D("genLeg1Eta",                         "genLeg1Eta",                              198,     -9.9,         +9.9);
  histogramGenLeg1Phi_                         = dqmStore.book1D("genLeg1Phi",                         "genLeg1Phi",                               72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg1X_                           = dqmStore.book1D("genLeg1X",                           "X_{1}^{gen}",                             102,     -0.01,         1.01);
  histogramGenLeg1XforGenLeg2X0_00to0_25_      = dqmStore.book1D("genLeg1XforGenLeg2X0_00to0_25",      "X_{1}^{gen} (0.00 < X_{2}^{gen} < 0.25)", 102,     -0.01,         1.01);
  histogramGenLeg1XforGenLeg2X0_25to0_50_      = dqmStore.book1D("genLeg1XforGenLeg2X0_25to0_50",      "X_{1}^{gen} (0.25 < X_{2}^{gen} < 0.50)", 102,     -0.01,         1.01);
  histogramGenLeg1XforGenLeg2X0_50to0_75_      = dqmStore.book1D("genLeg1XforGenLeg2X0_50to0_75",      "X_{1}^{gen} (0.50 < X_{2}^{gen} < 0.75)", 102,     -0.01,         1.01);
  histogramGenLeg1XforGenLeg2X0_75to1_00_      = dqmStore.book1D("genLeg1XforGenLeg2X0_75to1_00",      "X_{1}^{gen} (0.75 < X_{2}^{gen} < 1.00)", 102,     -0.01,         1.01);
  histogramGenLeg1Mt_                          = dqmStore.book1D("genLeg1Mt",                          "genLeg1Mt",                               250,      0.,         250.);
  histogramRecLeg1X_                           = dqmStore.book1D("recLeg1X",                           "recLeg1X",                                102,     -0.01,         1.01);
  histogramRecLeg1PFMt_                        = dqmStore.book1D("recLeg1PFMt",                        "recLeg1PFMt",                             250,      0.,         250.);
  histogramGenTau2Pt_                          = dqmStore.book1D("genTau2Pt",                          "genTau2Pt",                               250,      0.,         250.);
  histogramGenTau2Eta_                         = dqmStore.book1D("genTau2Eta",                         "genTau2Eta",                              198,     -9.9,         +9.9);
  histogramGenTau2Phi_                         = dqmStore.book1D("genTau2Phi",                         "genTau2Phi",                               72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg2Pt_                          = dqmStore.book1D("genLeg2Pt",                          "genLeg2Pt",                               250,      0.,         250.);
  histogramGenLeg2Eta_                         = dqmStore.book1D("genLeg2Eta",                         "genLeg2Eta",                              198,     -9.9,         +9.9);
  histogramGenLeg2Phi_                         = dqmStore.book1D("genLeg2Phi",                         "genLeg2Phi",                               72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg2X_                           = dqmStore.book1D("genLeg2X",                           "X_{2}^{gen}",                             102,     -0.01,         1.01);
  histogramGenLeg2XforGenLeg1X0_00to0_25_      = dqmStore.book1D("genLeg2XforGenLeg1X0_00to0_25",      "X_{2}^{gen} (0.00 < X_{1}^{gen} < 0.25)", 102,     -0.01,         1.01);
  histogramGenLeg2XforGenLeg1X0_25to0_50_      = dqmStore.book1D("genLeg2XforGenLeg1X0_25to0_50",      "X_{2}^{gen} (0.25 < X_{1}^{gen} < 0.50)", 102,     -0.01,         1.01);
  histogramGenLeg2XforGenLeg1X0_50to0_75_      = dqmStore.book1D("genLeg2XforGenLeg1X0_50to0_75",      "X_{2}^{gen} (0.50 < X_{1}^{gen} < 0.75)", 102,     -0.01,         1.01);
  histogramGenLeg2XforGenLeg1X0_75to1_00_      = dqmStore.book1D("genLeg2XforGenLeg1X0_75to1_00",      "X_{2}^{gen} (0.75 < X_{1}^{gen} < 1.00)", 102,     -0.01,         1.01);
  histogramGenLeg2Mt_                          = dqmStore.book1D("genLeg2Mt",                          "genLeg2Mt",                               250,      0.,         250.);
  histogramRecLeg2X_                           = dqmStore.book1D("recLeg2X",                           "recLeg2X",                                102,     -0.01,         1.01);
  histogramRecLeg2PFMt_                        = dqmStore.book1D("recLeg2PFMt",                        "recLeg2PFMt",                             250,      0.,         250.);

  histogramSumGenParticlePt_                   = dqmStore.book1D("sumGenParticlePt",                   "sumGenParticlePt",                        250,      0.,         250.);
  histogramSumGenParticlePt_charged_           = dqmStore.book1D("sumGenParticlePt_charged",           "sumGenParticlePt_charged",                250,      0.,         250.);
  histogramGenCaloMEt_                         = dqmStore.book1D("genCaloMEt",                         "genCaloMEt",                              250,      0.,         250.);
  histogramGenPFMEt_                           = dqmStore.book1D("genPFMEt",                           "genPFMEt",                                250,      0.,         250.);

  histogramRecCaloMEtECAL_                     = dqmStore.book1D("recCaloMEtECAL",                     "recCaloMEtECAL",                          250,      0.,         250.);
  histogramRecCaloSumEtECAL_                   = dqmStore.book1D("recCaloSumEtECAL",                   "recCaloSumEtECAL",                       2500,      0.,        2500.);
  histogramRecCaloMEtHCAL_                     = dqmStore.book1D("recCaloMEtHCAL",                     "recCaloMEtHCAL",                          250,      0.,         250.);
  histogramRecCaloSumEtHCAL_                   = dqmStore.book1D("recCaloSumEtHCAL",                   "recCaloSumEtHCAL",                       2500,      0.,        2500.);
  histogramRecCaloMEtHF_                       = dqmStore.book1D("recCaloMEtHF",                       "recCaloMEtHF",                            250,      0.,         250.);
  histogramRecCaloSumEtHF_                     = dqmStore.book1D("recCaloSumEtHF",                     "recCaloSumEtHF",                         2500,      0.,        2500.);
  histogramRecCaloMEtHO_                       = dqmStore.book1D("recCaloMEtHO",                       "recCaloMEtHO",                            250,      0.,         250.);  
  histogramRecCaloSumEtHO_                     = dqmStore.book1D("recCaloSumEtHO",                     "recCaloSumEtHO",                         2500,      0.,        2500.);

  // CV: Record presence in the embedded event of high Pt tracks, PFCandidates and muons 
  //     reconstructed near the direction of the replaced muons
  //    (indicating that maybe not all muon signals were removed in the embedding).
  //     Fill product of reconstructed track/PFCandidate/muon charge * charge of replaced muon into histogram.
  //     In case all matches happen just by chance, expect equal number of entries in positive and negative bins
  histogramWarning_recTrackNearReplacedMuon_   = dqmStore.book1D("Warning_recTrackNearReplacedMuon",   "Warning_recTrackNearReplacedMuon",          3,    -1.5,        +1.5);
  histogramWarning_recPFCandNearReplacedMuon_  = dqmStore.book1D("Warning_recPFCandNearReplacedMuon",  "Warning_recPFCandNearReplacedMuon",         3,    -1.5,        +1.5);
  histogramWarning_recMuonNearReplacedMuon_    = dqmStore.book1D("Warning_recMuonNearReplacedMuon",    "Warning_recMuonNearReplacedMuon",           3,    -1.5,        +1.5);

  for ( vInputTag::const_iterator srcWeight = srcOtherWeights_.begin();
	srcWeight != srcOtherWeights_.end(); ++srcWeight ) {
    plotEntryTypeEvtWeight* evtWeightPlotEntry = new plotEntryTypeEvtWeight(*srcWeight, dqmDirectory_);
    evtWeightPlotEntry->bookHistograms(dqmStore);
    evtWeightPlotEntries_.push_back(evtWeightPlotEntry);
  }
  if ( srcMuonRadCorrWeight_.label() != "" ) {
    plotEntryTypeEvtWeight* evtWeightPlotEntry = new plotEntryTypeEvtWeight(srcMuonRadCorrWeight_, dqmDirectory_);
    evtWeightPlotEntry->bookHistograms(dqmStore);
    evtWeightPlotEntries_.push_back(evtWeightPlotEntry);
  }

  typedef std::pair<int, int> pint;
  std::vector<pint> jetBins;
  jetBins.push_back(pint(-1, -1));
  jetBins.push_back(pint(0, 0));
  jetBins.push_back(pint(1, 1));
  jetBins.push_back(pint(2, 2));
  jetBins.push_back(pint(3, 1000));
  for ( std::vector<pint>::const_iterator jetBin = jetBins.begin();
	jetBin != jetBins.end(); ++jetBin ) {
    TString dqmDirectory_beforeRad = dqmDirectory_;
    if ( !dqmDirectory_beforeRad.EndsWith("/") ) dqmDirectory_beforeRad.Append("/");
    dqmDirectory_beforeRad.Append("beforeRad");
    plotEntryTypeMuonRadCorrUncertainty* muonRadCorrUncertaintyPlotEntry_beforeRad = new plotEntryTypeMuonRadCorrUncertainty(jetBin->first, jetBin->second, dqmDirectory_beforeRad.Data());
    muonRadCorrUncertaintyPlotEntry_beforeRad->bookHistograms(dqmStore);
    muonRadCorrUncertaintyPlotEntries_beforeRad_.push_back(muonRadCorrUncertaintyPlotEntry_beforeRad);
    TString dqmDirectory_afterRad = dqmDirectory_;
    if ( !dqmDirectory_afterRad.EndsWith("/") ) dqmDirectory_afterRad.Append("/");
    dqmDirectory_afterRad.Append("afterRad");
    plotEntryTypeMuonRadCorrUncertainty* muonRadCorrUncertaintyPlotEntry_afterRad = new plotEntryTypeMuonRadCorrUncertainty(jetBin->first, jetBin->second, dqmDirectory_afterRad.Data());
    muonRadCorrUncertaintyPlotEntry_afterRad->bookHistograms(dqmStore);
    muonRadCorrUncertaintyPlotEntries_afterRad_.push_back(muonRadCorrUncertaintyPlotEntry_afterRad);
    TString dqmDirectory_afterRadAndCorr = dqmDirectory_;
    if ( !dqmDirectory_afterRadAndCorr.EndsWith("/") ) dqmDirectory_afterRadAndCorr.Append("/");
    dqmDirectory_afterRadAndCorr.Append("afterRadAndCorr");
    plotEntryTypeMuonRadCorrUncertainty* muonRadCorrUncertaintyPlotEntry_afterRadAndCorr = new plotEntryTypeMuonRadCorrUncertainty(jetBin->first, jetBin->second, dqmDirectory_afterRadAndCorr.Data());
    muonRadCorrUncertaintyPlotEntry_afterRadAndCorr->bookHistograms(dqmStore);
    muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_.push_back(muonRadCorrUncertaintyPlotEntry_afterRadAndCorr);
  }
  muonRadCorrUncertainty_numWarnings_ = 0;
  muonRadCorrUncertainty_maxWarnings_ = 3;

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
  bookHistograms(electronDistributionsExtra_, dqmStore);
  bookHistograms(electronEfficiencies_, dqmStore);
  bookHistograms(gsfElectronEfficiencies_, dqmStore);
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
    //std::cout << "<fillVisPtEtaPhiMassDistributions>:" << std::endl;
    //std::cout << " srcLeg1 = " << srcLeg1 << std::endl;
    //std::cout << " srcLeg2 = " << srcLeg2 << std::endl;
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
	  //std::cout << "leg1: Pt = " << visDecayProduct1->pt() << ", phi = " << visDecayProduct1->phi() << " (Px = " << visDecayProduct1->px() << ", Py = " << visDecayProduct1->py() << ")" << std::endl;
	  //std::cout << "leg2: Pt = " << visDecayProduct2->pt() << ", phi = " << visDecayProduct2->phi() << " (Px = " << visDecayProduct2->px() << ", Py = " << visDecayProduct2->py() << ")" << std::endl;
	  reco::Candidate::LorentzVector visDiTauP4 = visDecayProduct1->p4() + visDecayProduct2->p4();
	  histogram_visDiTauPt->Fill(visDiTauP4.pt(), evtWeight);
	  histogram_visDiTauEta->Fill(visDiTauP4.eta(), evtWeight);
	  histogram_visDiTauPhi->Fill(visDiTauP4.phi(), evtWeight);
	  histogram_visDiTauMass->Fill(visDiTauP4.mass(), evtWeight);
	}
      }
    }
  }

  double compMt(const reco::Candidate::LorentzVector& visP4, const reco::Candidate::LorentzVector& metP4)
  {
    double sumPx = visP4.px() + metP4.px();
    double sumPy = visP4.py() + metP4.py();
    double sumEt = TMath::Max(visP4.Et(), visP4.pt()) + TMath::Sqrt(metP4.pt());
    double mt2 = sumEt*sumEt - (sumPx*sumPx + sumPy*sumPy);
    if ( mt2 < 0 ) mt2 = 0.;
    return TMath::Sqrt(mt2);
  }
  
  void fillX1andX2Distributions(const edm::Event& evt, 
				const edm::InputTag& srcGenDiTau, const edm::InputTag& srcLeg1, const edm::InputTag& srcLeg2, const reco::Candidate::LorentzVector& metP4,
				MonitorElement* histogram_tau1Pt, MonitorElement* histogram_tau1Eta, MonitorElement* histogram_tau1Phi, 
				MonitorElement* histogram_leg1Pt, MonitorElement* histogram_leg1Eta, MonitorElement* histogram_leg1Phi, 
				MonitorElement* histogram_leg1X, 
				MonitorElement* histogram_leg1XforLeg2X0_00to0_25, 
				MonitorElement* histogram_leg1XforLeg2X0_25to0_50, 
				MonitorElement* histogram_leg1XforLeg2X0_50to0_75, 
				MonitorElement* histogram_leg1XforLeg2X0_75to1_00, 
				MonitorElement* histogram_leg1Mt, 
				MonitorElement* histogram_tau2Pt, MonitorElement* histogram_tau2Eta, MonitorElement* histogram_tau2Phi, 
				MonitorElement* histogram_leg2Pt, MonitorElement* histogram_leg2Eta, MonitorElement* histogram_leg2Phi, 
				MonitorElement* histogram_leg2X, 
				MonitorElement* histogram_leg2XforLeg1X0_00to0_25, 
				MonitorElement* histogram_leg2XforLeg1X0_25to0_50, 
				MonitorElement* histogram_leg2XforLeg1X0_50to0_75, 
				MonitorElement* histogram_leg2XforLeg1X0_75to1_00, 
				MonitorElement* histogram_leg2Mt, 
				MonitorElement* histogram_VisDeltaPhiLeg1Leg2,
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
	    if ( histogram_tau1Pt && histogram_tau1Eta && histogram_tau1Phi ) {
	      histogram_tau1Pt->Fill(genLeg1->pt(), evtWeight);
	      histogram_tau1Eta->Fill(genLeg1->eta(), evtWeight);
	      histogram_tau1Phi->Fill(genLeg1->phi(), evtWeight);
	    }
	    if ( histogram_leg1Pt && histogram_leg1Eta && histogram_leg1Phi ) {
	      histogram_leg1Pt->Fill(visDecayProduct1->pt(), evtWeight);
	      histogram_leg1Eta->Fill(visDecayProduct1->eta(), evtWeight);
	      histogram_leg1Phi->Fill(visDecayProduct1->phi(), evtWeight);
	    }
	    histogram_leg1X->Fill(X1, evtWeight);
	    if ( histogram_leg1XforLeg2X0_00to0_25 && histogram_leg1XforLeg2X0_25to0_50 && histogram_leg1XforLeg2X0_50to0_75 && histogram_leg1XforLeg2X0_75to1_00 ) {
	      if      ( X2 < 0.25 ) histogram_leg1XforLeg2X0_00to0_25->Fill(X1, evtWeight);
	      else if ( X2 < 0.50 ) histogram_leg1XforLeg2X0_25to0_50->Fill(X1, evtWeight);
	      else if ( X2 < 0.75 ) histogram_leg1XforLeg2X0_50to0_75->Fill(X1, evtWeight);
	      else                  histogram_leg1XforLeg2X0_75to1_00->Fill(X1, evtWeight);
	    }
	    if ( histogram_leg1Mt ) histogram_leg1Mt->Fill(compMt(visDecayProduct1->p4(), metP4), evtWeight);
	    if ( histogram_tau2Pt && histogram_tau2Eta && histogram_tau2Phi ) {
	      histogram_tau2Pt->Fill(genLeg2->pt(), evtWeight);
	      histogram_tau2Eta->Fill(genLeg2->eta(), evtWeight);
	      histogram_tau2Phi->Fill(genLeg2->phi(), evtWeight);
	    }
	    if ( histogram_leg2Pt && histogram_leg2Eta && histogram_leg2Phi ) {
	      histogram_leg2Pt->Fill(visDecayProduct2->pt(), evtWeight);
	      histogram_leg2Eta->Fill(visDecayProduct2->eta(), evtWeight);
	      histogram_leg2Phi->Fill(visDecayProduct2->phi(), evtWeight);
	    }
	    histogram_leg2X->Fill(X2, evtWeight);
	    if ( histogram_leg2XforLeg1X0_00to0_25 && histogram_leg2XforLeg1X0_25to0_50 && histogram_leg2XforLeg1X0_50to0_75 && histogram_leg2XforLeg1X0_75to1_00 ) {
	      if      ( X1 < 0.25 ) histogram_leg2XforLeg1X0_00to0_25->Fill(X2, evtWeight);
	      else if ( X1 < 0.50 ) histogram_leg2XforLeg1X0_25to0_50->Fill(X2, evtWeight);
	      else if ( X1 < 0.75 ) histogram_leg2XforLeg1X0_50to0_75->Fill(X2, evtWeight);
	      else                  histogram_leg2XforLeg1X0_75to1_00->Fill(X2, evtWeight);
	      if ( histogram_leg2Mt ) histogram_leg2Mt->Fill(compMt(visDecayProduct2->p4(), metP4), evtWeight);
	    }
	    histogram_VisDeltaPhiLeg1Leg2->Fill(normalizedPhi(visDecayProduct1->phi() - visDecayProduct2->phi()), evtWeight);
	  }
	}
      }  
    }
  }
}

namespace
{
  std::string runLumiEventNumbers_to_string(const edm::Event& evt)
  {
    edm::RunNumber_t run_number = evt.id().run();
    edm::LuminosityBlockNumber_t ls_number = evt.luminosityBlock();
    edm::EventNumber_t event_number = evt.id().event();
    std::ostringstream retVal;
    retVal << "Run = " << run_number << ", LS = " << ls_number << ", Event = " << event_number;
    return retVal.str();
  }
}

void MCEmbeddingValidationAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<MCEmbeddingValidationAnalyzer::analyze>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  const reco::Candidate* replacedMuonPlus  = 0;
  const reco::Candidate* replacedMuonMinus = 0;
  if ( srcReplacedMuons_.label() != "" ) {
    std::vector<reco::CandidateBaseRef> replacedMuons = getSelMuons(evt, srcReplacedMuons_);
    for ( std::vector<reco::CandidateBaseRef>::const_iterator replacedMuon = replacedMuons.begin();
	  replacedMuon != replacedMuons.end(); ++replacedMuon ) {
      if      ( (*replacedMuon)->charge() > +0.5 ) replacedMuonPlus  = replacedMuon->get();
      else if ( (*replacedMuon)->charge() < -0.5 ) replacedMuonMinus = replacedMuon->get();
    }
  }
  if ( verbosity_ ) {
    if ( replacedMuonPlus ) std::cout << "replacedMuonPlus: Pt = " << replacedMuonPlus->pt() << ", eta = " << replacedMuonPlus->eta() << ", phi = " << replacedMuonPlus->phi() << std::endl;
    if ( replacedMuonMinus ) std::cout << "replacedMuonMinus: Pt = " << replacedMuonMinus->pt() << ", eta = " << replacedMuonMinus->eta() << ", phi = " << replacedMuonMinus->phi() << std::endl;
  }
  
  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> genDiTaus;
  evt.getByLabel(srcGenDiTaus_, genDiTaus);

  const reco::Candidate* genTauPlus  = 0;
  const reco::Candidate* genTauMinus = 0;
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
    if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) continue;
    size_t numDaughters = genDiTau_composite->numberOfDaughters();
    for ( size_t iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
      const reco::Candidate* daughter = genDiTau_composite->daughter(iDaughter);
      if      ( daughter->charge() > +0.5 ) genTauPlus  = daughter;
      else if ( daughter->charge() < -0.5 ) genTauMinus = daughter;
    }
  }
  if ( verbosity_ ) {
    if ( genTauPlus ) std::cout << "genTauPlus: Pt = " << genTauPlus->pt() << ", eta = " << genTauPlus->eta() << ", phi = " << genTauPlus->phi() << std::endl;
    if ( genTauMinus ) std::cout << "genTauMinus: Pt = " << genTauMinus->pt() << ", eta = " << genTauMinus->eta() << ", phi = " << genTauMinus->phi() << std::endl;
  }

  double dRmin = 1.e+3;
  if ( replacedMuonPlus && genTauPlus ) {
    double dR = deltaR(genTauPlus->p4(),  replacedMuonPlus->p4());
    if ( verbosity_ ) std::cout << " dR(replacedMuonPlus, genTauPlus) = " << dR << std::endl;
    if ( dR < dRmin ) dRmin = dR;
  }
  if ( replacedMuonPlus && genTauMinus ) {
    double dR = deltaR(genTauMinus->p4(),  replacedMuonPlus->p4());
    if ( verbosity_ ) std::cout << " dR(replacedMuonPlus, genTauMinus) = " << dR << std::endl;
    if ( dR < dRmin ) dRmin = dR;
  }
  if ( replacedMuonMinus && genTauPlus ) {
    double dR = deltaR(genTauPlus->p4(),  replacedMuonMinus->p4());
    if ( verbosity_ ) std::cout << " dR(replacedMuonMinus, genTauPlus) = " << dR << std::endl;
    if ( dR < dRmin ) dRmin = dR;
  }
  if ( replacedMuonMinus && genTauMinus ) {
    double dR = deltaR(genTauMinus->p4(),  replacedMuonMinus->p4());
    if ( verbosity_ ) std::cout << " dR(replacedMuonMinus, genTauMinus) = " << dR << std::endl;
    if ( dR < dRmin ) dRmin = dR;
  }
  if ( verbosity_ ) std::cout << "--> dRmin = " << dRmin << std::endl;
  if ( dRmin < dRminSeparation_ ) return;
  if ( verbosity_ ) std::cout << " cut on dRminSeparation = " << dRminSeparation_ << " passed." << std::endl;

//--- compute event weight
  double evtWeight = 1.0;
  std::map<std::string, double> evtWeightMap;
  for ( vInputTag::const_iterator srcWeight = srcOtherWeights_.begin();
	srcWeight != srcOtherWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    //std::cout << "weight(" << srcWeight->label().data() << ":" << srcWeight->instance().data() << ") = " << (*weight) << std::endl;
    evtWeight *= (*weight);
    evtWeightMap[Form("%s_%s", srcWeight->label().data(), srcWeight->instance().data())] = (*weight);
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
      evtWeightMap[Form("%s_%s", srcGenFilterInfo_.label().data(), srcGenFilterInfo_.instance().data())] = weight;
    }
  }

  if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) return;

  histogramEventCounter_->Fill(1., evtWeight);

  double muonRadCorrWeight     = 1.;
  double muonRadCorrWeightUp   = 1.;
  double muonRadCorrWeightDown = 1.;
  if ( srcMuonRadCorrWeight_.label() != "" && srcMuonRadCorrWeightUp_.label() != "" && srcMuonRadCorrWeightDown_.label() != "" ) {
    edm::Handle<double> muonRadCorrWeight_handle;
    evt.getByLabel(srcMuonRadCorrWeight_, muonRadCorrWeight_handle);
    muonRadCorrWeight = (*muonRadCorrWeight_handle);
    evtWeightMap["muonRadCorrWeight"] = muonRadCorrWeight;
    edm::Handle<double> muonRadCorrWeightUp_handle;
    evt.getByLabel(srcMuonRadCorrWeightUp_, muonRadCorrWeightUp_handle);
    muonRadCorrWeightUp = (*muonRadCorrWeightUp_handle);
    edm::Handle<double> muonRadCorrWeightDown_handle;
    evt.getByLabel(srcMuonRadCorrWeightDown_, muonRadCorrWeightDown_handle);
    muonRadCorrWeightDown = (*muonRadCorrWeightDown_handle);
  }
  //std::cout << " muonRadCorrWeight = " << muonRadCorrWeight 
  //	      << " + " << (muonRadCorrWeightUp - muonRadCorrWeight)
  //	      << " - " << (muonRadCorrWeight - muonRadCorrWeightDown) << std::endl;
  
  for ( std::vector<plotEntryTypeEvtWeight*>::iterator evtWeightPlotEntry = evtWeightPlotEntries_.begin();
	evtWeightPlotEntry != evtWeightPlotEntries_.end(); ++evtWeightPlotEntry ) {
    (*evtWeightPlotEntry)->fillHistograms(evt, evtWeightMap);
  }
  
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
  histogramNumTracksPtGt5_->Fill(numTracksPtGt5, muonRadCorrWeight*evtWeight);
  histogramNumTracksPtGt10_->Fill(numTracksPtGt10, muonRadCorrWeight*evtWeight);
  histogramNumTracksPtGt20_->Fill(numTracksPtGt20, muonRadCorrWeight*evtWeight);
  histogramNumTracksPtGt30_->Fill(numTracksPtGt30, muonRadCorrWeight*evtWeight);
  histogramNumTracksPtGt40_->Fill(numTracksPtGt40, muonRadCorrWeight*evtWeight);
  
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
  histogramNumChargedPFCandsPtGt5_->Fill(numChargedPFCandsPtGt5, muonRadCorrWeight*evtWeight);
  histogramNumChargedPFCandsPtGt10_->Fill(numChargedPFCandsPtGt10, muonRadCorrWeight*evtWeight);
  histogramNumChargedPFCandsPtGt20_->Fill(numChargedPFCandsPtGt20, muonRadCorrWeight*evtWeight);
  histogramNumChargedPFCandsPtGt30_->Fill(numChargedPFCandsPtGt30, muonRadCorrWeight*evtWeight);
  histogramNumChargedPFCandsPtGt40_->Fill(numChargedPFCandsPtGt40, muonRadCorrWeight*evtWeight);
  histogramNumNeutralPFCandsPtGt5_->Fill(numNeutralPFCandsPtGt5, muonRadCorrWeight*evtWeight);
  histogramNumNeutralPFCandsPtGt10_->Fill(numNeutralPFCandsPtGt10, muonRadCorrWeight*evtWeight);
  histogramNumNeutralPFCandsPtGt20_->Fill(numNeutralPFCandsPtGt20, muonRadCorrWeight*evtWeight);
  histogramNumNeutralPFCandsPtGt30_->Fill(numNeutralPFCandsPtGt30, muonRadCorrWeight*evtWeight);
  histogramNumNeutralPFCandsPtGt40_->Fill(numNeutralPFCandsPtGt40, muonRadCorrWeight*evtWeight);

  edm::Handle<pat::JetCollection> jets;
  evt.getByLabel(srcRecJets_, jets);
  int numJetsRawPtGt20                = 0;
  int numJetsRawPtGt20AbsEtaLt2_5     = 0;
  int numJetsRawPtGt20AbsEta2_5to4_5  = 0;
  int numJetsCorrPtGt20               = 0;
  int numJetsCorrPtGt20AbsEtaLt2_5    = 0;
  int numJetsCorrPtGt20AbsEta2_5to4_5 = 0;
  int numJetsRawPtGt30                = 0;
  int numJetsRawPtGt30AbsEtaLt2_5     = 0;
  int numJetsRawPtGt30AbsEta2_5to4_5  = 0;
  int numJetsCorrPtGt30               = 0;
  int numJetsCorrPtGt30AbsEtaLt2_5    = 0;
  int numJetsCorrPtGt30AbsEta2_5to4_5 = 0;
  for ( pat::JetCollection::const_iterator jet = jets->begin();
	jet != jets->end(); ++jet ) {

    reco::Candidate::LorentzVector rawJetP4 = jet->correctedP4("Uncorrected");
    double rawJetPt = rawJetP4.pt();
    double rawJetAbsEta = TMath::Abs(rawJetP4.eta());
    if ( rawJetAbsEta < 4.5 ) { // CV: do not consider any jet reconstructed outside eta range used in H -> tautau analysis
      histogramRawJetPt_->Fill(rawJetPt, muonRadCorrWeight*evtWeight);
      if      ( rawJetAbsEta < 2.5 ) histogramRawJetPtAbsEtaLt2_5_->Fill(rawJetPt, muonRadCorrWeight*evtWeight);
      else if ( rawJetAbsEta < 4.5 ) histogramRawJetPtAbsEta2_5to4_5_->Fill(rawJetPt, muonRadCorrWeight*evtWeight);
      if ( rawJetPt > 20. ) {
	histogramRawJetEtaPtGt20_->Fill(rawJetP4.eta(), muonRadCorrWeight*evtWeight);
	++numJetsRawPtGt20;
	if      ( rawJetAbsEta < 2.5 ) ++numJetsRawPtGt20AbsEtaLt2_5;
	else if ( rawJetAbsEta < 4.5 ) ++numJetsRawPtGt20AbsEta2_5to4_5;
      }
      if ( rawJetPt > 30. ) {
	histogramRawJetEtaPtGt20_->Fill(rawJetP4.eta(), muonRadCorrWeight*evtWeight);
	++numJetsRawPtGt30;
	if      ( rawJetAbsEta < 2.5 ) ++numJetsRawPtGt30AbsEtaLt2_5;
	else if ( rawJetAbsEta < 4.5 ) ++numJetsRawPtGt30AbsEta2_5to4_5;
      }
    }

    reco::Candidate::LorentzVector corrJetP4 = jet->p4();
    double corrJetPt = corrJetP4.pt();
    double corrJetAbsEta = TMath::Abs(corrJetP4.eta());
    if ( corrJetAbsEta < 4.5 ) { // CV: do not consider any jet reconstructed outside eta range used in H -> tautau analysis
      histogramCorrJetPt_->Fill(corrJetPt, muonRadCorrWeight*evtWeight);
      if      ( corrJetAbsEta < 2.5 ) histogramCorrJetPtAbsEtaLt2_5_->Fill(corrJetPt, muonRadCorrWeight*evtWeight);
      else if ( corrJetAbsEta < 4.5 ) histogramCorrJetPtAbsEta2_5to4_5_->Fill(corrJetPt, muonRadCorrWeight*evtWeight);
      if ( corrJetPt > 20. ) {
	histogramCorrJetEtaPtGt20_->Fill(corrJetP4.eta(), muonRadCorrWeight*evtWeight);
	++numJetsCorrPtGt20;
	if      ( corrJetAbsEta < 2.5 ) ++numJetsCorrPtGt20AbsEtaLt2_5;
	else if ( corrJetAbsEta < 4.5 ) ++numJetsCorrPtGt20AbsEta2_5to4_5;
      }
      if ( corrJetPt > 30. ) {
	histogramCorrJetEtaPtGt20_->Fill(corrJetP4.eta(), muonRadCorrWeight*evtWeight);
	++numJetsCorrPtGt30;
	if      ( corrJetAbsEta < 2.5 ) ++numJetsCorrPtGt30AbsEtaLt2_5;
	else if ( corrJetAbsEta < 4.5 ) ++numJetsCorrPtGt30AbsEta2_5to4_5;
      }
    }    
  }
  histogramNumJetsRawPtGt20_->Fill(numJetsRawPtGt20, muonRadCorrWeight*evtWeight);
  histogramNumJetsRawPtGt20AbsEtaLt2_5_->Fill(numJetsRawPtGt20AbsEtaLt2_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsRawPtGt20AbsEta2_5to4_5_->Fill(numJetsRawPtGt20AbsEta2_5to4_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt20_->Fill(numJetsCorrPtGt20, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt20AbsEtaLt2_5_->Fill(numJetsCorrPtGt20AbsEtaLt2_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt20AbsEta2_5to4_5_->Fill(numJetsCorrPtGt20AbsEta2_5to4_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsRawPtGt30_->Fill(numJetsRawPtGt30, muonRadCorrWeight*evtWeight);
  histogramNumJetsRawPtGt30AbsEtaLt2_5_->Fill(numJetsRawPtGt30AbsEtaLt2_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsRawPtGt30AbsEta2_5to4_5_->Fill(numJetsRawPtGt30AbsEta2_5to4_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt30_->Fill(numJetsCorrPtGt30, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt30AbsEtaLt2_5_->Fill(numJetsCorrPtGt30AbsEtaLt2_5, muonRadCorrWeight*evtWeight);
  histogramNumJetsCorrPtGt30AbsEta2_5to4_5_->Fill(numJetsCorrPtGt30AbsEta2_5to4_5, muonRadCorrWeight*evtWeight);

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
  histogramNumGlobalMuons_->Fill(numGlobalMuons, muonRadCorrWeight*evtWeight);
  histogramNumStandAloneMuons_->Fill(numStandAloneMuons, muonRadCorrWeight*evtWeight);
  histogramNumPFMuons_->Fill(numPFMuons, muonRadCorrWeight*evtWeight);

  edm::Handle<reco::VertexCollection> theVertex;
  evt.getByLabel(srcTheRecVertex_, theVertex);
  if ( theVertex->size() >= 1 ) {
    const reco::Vertex::Point& theVertexPosition = theVertex->front().position();
    histogramTheRecVertexX_->Fill(theVertexPosition.x(), muonRadCorrWeight*evtWeight);
    histogramTheRecVertexY_->Fill(theVertexPosition.y(), muonRadCorrWeight*evtWeight);
    histogramTheRecVertexZ_->Fill(theVertexPosition.z(), muonRadCorrWeight*evtWeight);
  }
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByLabel(srcRecVertices_, vertices);
  for ( reco::VertexCollection::const_iterator vertex = vertices->begin();
	vertex != vertices->end(); ++vertex ) {
    histogramRecVertexX_->Fill(vertex->position().x(), muonRadCorrWeight*evtWeight);
    histogramRecVertexY_->Fill(vertex->position().y(), muonRadCorrWeight*evtWeight);
    histogramRecVertexZ_->Fill(vertex->position().z(), muonRadCorrWeight*evtWeight);
  }
  histogramNumRecVertices_->Fill(vertices->size(), muonRadCorrWeight*evtWeight);
  edm::Handle<reco::VertexCollection> verticesWithBS;
  evt.getByLabel(srcRecVerticesWithBS_, verticesWithBS);
  for ( reco::VertexCollection::const_iterator vertex = verticesWithBS->begin();
	vertex != verticesWithBS->end(); ++vertex ) {
    histogramRecVertexWithBSx_->Fill(vertex->position().x(), muonRadCorrWeight*evtWeight);
    histogramRecVertexWithBSy_->Fill(vertex->position().y(), muonRadCorrWeight*evtWeight);
    histogramRecVertexWithBSz_->Fill(vertex->position().z(), muonRadCorrWeight*evtWeight);
  }
  histogramNumRecVerticesWithBS_->Fill(verticesWithBS->size(), muonRadCorrWeight*evtWeight);
  
  edm::Handle<reco::BeamSpot> beamSpot;
  evt.getByLabel(srcBeamSpot_, beamSpot);
  if ( beamSpot.isValid() ) { 
    histogramBeamSpotX_->Fill(beamSpot->position().x(), muonRadCorrWeight*evtWeight);
    histogramBeamSpotY_->Fill(beamSpot->position().y(), muonRadCorrWeight*evtWeight);
  }
      
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    histogramGenDiTauPt_->Fill(genDiTau->pt(), muonRadCorrWeight*evtWeight);
    histogramGenDiTauEta_->Fill(genDiTau->eta(), muonRadCorrWeight*evtWeight);
    histogramGenDiTauPhi_->Fill(genDiTau->phi(), muonRadCorrWeight*evtWeight);
    histogramGenDiTauMass_->Fill(genDiTau->mass(), muonRadCorrWeight*evtWeight);
  }

  bool passesCutsBeforeRotation = false;
  if ( (replacedMuonPlus  && replacedMuonPlus->pt()  > replacedMuonPtThresholdHigh_ && replacedMuonMinus && replacedMuonMinus->pt() > replacedMuonPtThresholdLow_) ||
       (replacedMuonMinus && replacedMuonMinus->pt() > replacedMuonPtThresholdHigh_ && replacedMuonPlus  && replacedMuonPlus->pt()  > replacedMuonPtThresholdLow_) ) passesCutsBeforeRotation = true;
  bool passesCutsAfterRotation = false;
  for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	genDiTau != genDiTaus->end(); ++genDiTau ) {
    const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
    if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) continue;
    const reco::Candidate* genTau1 = genDiTau_composite->daughter(0);
    const reco::Candidate* genTau2 = genDiTau_composite->daughter(1);
    if ( !(genTau1 && genTau2) ) continue;
    if ( (genTau1->pt() > replacedMuonPtThresholdHigh_ && genTau2->pt() > replacedMuonPtThresholdLow_ ) ||
	 (genTau1->pt() > replacedMuonPtThresholdLow_  && genTau2->pt() > replacedMuonPtThresholdHigh_) ) {
      passesCutsAfterRotation = true;
      break;
    }
    histogramRotationAngleMatrix_->Fill(passesCutsBeforeRotation, passesCutsAfterRotation, muonRadCorrWeight*evtWeight);   
  }
  if ( genTauPlus && genTauMinus ) {
    histogramGenDeltaPhiLeg1Leg2_->Fill(normalizedPhi(genTauPlus->phi() - genTauMinus->phi()), muonRadCorrWeight*evtWeight);
    reco::Particle::LorentzVector diTauP4_lab = genTauPlus->p4() + genTauMinus->p4();
    ROOT::Math::Boost boost_to_rf(diTauP4_lab.BoostToCM());
    reco::Particle::LorentzVector diTauP4_rf = boost_to_rf(diTauP4_lab);
    reco::Particle::LorentzVector tauPlusP4_rf = boost_to_rf(genTauPlus->p4());
    if ( (diTauP4_rf.P()*tauPlusP4_rf.P()) > 0. ) {
      double cosGjAngle = (diTauP4_rf.px()*tauPlusP4_rf.px() + diTauP4_rf.py()*tauPlusP4_rf.py() + diTauP4_rf.pz()*tauPlusP4_rf.pz())/(diTauP4_rf.P()*tauPlusP4_rf.P());
      double gjAngle = TMath::ACos(cosGjAngle);
      histogramGenDiTauDecayAngle_->Fill(gjAngle, muonRadCorrWeight*evtWeight);
    }
  }
  if ( replacedMuonPlus && genTauPlus && replacedMuonMinus && genTauMinus ) {
    histogramRotationLegPlusDeltaR_->Fill(deltaR(genTauPlus->p4(), replacedMuonPlus->p4()), muonRadCorrWeight*evtWeight);
    histogramRotationLegMinusDeltaR_->Fill(deltaR(genTauMinus->p4(), replacedMuonMinus->p4()), muonRadCorrWeight*evtWeight);
    
    reco::Particle::LorentzVector diTauP4_lab = genTauPlus->p4() + genTauMinus->p4();
    histogramPhiRotLegPlus_->Fill(SVfit_namespace::phiLabFromLabMomenta(replacedMuonPlus->p4(), diTauP4_lab), muonRadCorrWeight*evtWeight);
    histogramPhiRotLegMinus_->Fill(SVfit_namespace::phiLabFromLabMomenta(replacedMuonMinus->p4(), diTauP4_lab), muonRadCorrWeight*evtWeight);
  }
  
  fillVisPtEtaPhiMassDistributions(evt, srcGenLeg1_, srcGenLeg2_, histogramGenVisDiTauPt_, histogramGenVisDiTauEta_, histogramGenVisDiTauPhi_, histogramGenVisDiTauMass_, muonRadCorrWeight*evtWeight);
  fillVisPtEtaPhiMassDistributions(evt, srcRecLeg1_, srcRecLeg2_, histogramRecVisDiTauPt_, histogramRecVisDiTauEta_, histogramRecVisDiTauPhi_, histogramRecVisDiTauMass_, muonRadCorrWeight*evtWeight);

  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> genCaloMETs;
  evt.getByLabel(srcGenCaloMEt_, genCaloMETs);
  const reco::Candidate::LorentzVector& genCaloMEtP4 = genCaloMETs->front().p4();
  histogramGenCaloMEt_->Fill(genCaloMEtP4.pt(), muonRadCorrWeight*evtWeight);
  edm::Handle<METView> genPFMETs;
  evt.getByLabel(srcGenPFMEt_, genPFMETs);
  const reco::Candidate::LorentzVector& genPFMEtP4 = genPFMETs->front().p4();
  histogramGenPFMEt_->Fill(genPFMEtP4.pt(), muonRadCorrWeight*evtWeight);

  fillX1andX2Distributions(evt, srcGenDiTaus_, srcGenLeg1_, srcGenLeg2_, genPFMEtP4,
			   histogramGenTau1Pt_, histogramGenTau1Eta_, histogramGenTau1Phi_,
			   histogramGenLeg1Pt_, histogramGenLeg1Eta_, histogramGenLeg1Phi_, histogramGenLeg1X_, 
			   histogramGenLeg1XforGenLeg2X0_00to0_25_, histogramGenLeg1XforGenLeg2X0_25to0_50_, histogramGenLeg1XforGenLeg2X0_50to0_75_, histogramGenLeg1XforGenLeg2X0_75to1_00_,
			   histogramGenLeg1Mt_, 
			   histogramGenTau2Pt_, histogramGenTau2Eta_, histogramGenTau2Phi_,
			   histogramGenLeg2Pt_, histogramGenLeg2Eta_, histogramGenLeg2Phi_, histogramGenLeg2X_, 
			   histogramGenLeg2XforGenLeg1X0_00to0_25_, histogramGenLeg2XforGenLeg1X0_25to0_50_, histogramGenLeg2XforGenLeg1X0_50to0_75_, histogramGenLeg2XforGenLeg1X0_75to1_00_,
			   histogramGenLeg2Mt_, 
			   histogramGenVisDeltaPhiLeg1Leg2_, muonRadCorrWeight*evtWeight);

  edm::Handle<METView> recPFMETs;
  evt.getByLabel(srcRecPFMEt_, recPFMETs);
  const reco::Candidate::LorentzVector& recPFMEtP4 = recPFMETs->front().p4();
  
  fillX1andX2Distributions(evt, srcGenDiTaus_, srcRecLeg1_, srcRecLeg2_, recPFMEtP4,
			   0, 0, 0, 
			   0, 0, 0, histogramRecLeg1X_, 
			   0, 0, 0, 0, 
			   histogramRecLeg1PFMt_, 
			   0, 0, 0, 
			   0, 0, 0, histogramRecLeg2X_, 
			   0, 0, 0, 0, histogramRecLeg2PFMt_, 
			   histogramRecVisDeltaPhiLeg1Leg2_, muonRadCorrWeight*evtWeight);
  
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
  histogramSumGenParticlePt_->Fill(sumGenParticleP4.pt(), muonRadCorrWeight*evtWeight);
  
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
  histogramRecCaloMEtECAL_->Fill(sumCaloTowerP4_ecal.pt(), muonRadCorrWeight*evtWeight);
  histogramRecCaloSumEtECAL_->Fill(sumEtCaloTowersECAL, muonRadCorrWeight*evtWeight);
  histogramRecCaloMEtHCAL_->Fill(sumCaloTowerP4_hcal.pt(), muonRadCorrWeight*evtWeight);
  histogramRecCaloSumEtHCAL_->Fill(sumEtCaloTowersHCAL, muonRadCorrWeight*evtWeight);
  histogramRecCaloMEtHF_->Fill(sumCaloTowerP4_hf.pt(), muonRadCorrWeight*evtWeight);
  histogramRecCaloSumEtHF_->Fill(sumEtCaloTowersHF, muonRadCorrWeight*evtWeight);
  histogramRecCaloMEtHO_->Fill(sumCaloTowerP4_ho.pt(), muonRadCorrWeight*evtWeight);
  histogramRecCaloSumEtHO_->Fill(sumEtCaloTowersHO, muonRadCorrWeight*evtWeight);
  
  if ( srcMuonsBeforeRad_.label() != "" && srcMuonsAfterRad_.label() != "" ) {
    reco::Candidate::LorentzVector genMuonPlusP4_beforeRad;
    bool genMuonPlus_beforeRad_found = false;
    reco::Candidate::LorentzVector genMuonMinusP4_beforeRad;
    bool genMuonMinus_beforeRad_found = false;
    reco::Candidate::LorentzVector genMuonPlusP4_afterRad;
    bool genMuonPlus_afterRad_found = false;
    reco::Candidate::LorentzVector genMuonMinusP4_afterRad;
    bool genMuonMinus_afterRad_found = false;
    
    findMuons(evt, srcMuonsBeforeRad_, genMuonPlusP4_beforeRad, genMuonPlus_beforeRad_found, genMuonMinusP4_beforeRad, genMuonMinus_beforeRad_found);
    findMuons(evt, srcMuonsAfterRad_, genMuonPlusP4_afterRad, genMuonPlus_afterRad_found, genMuonMinusP4_afterRad, genMuonMinus_afterRad_found);
 
    bool genMuonPlus_found = (genMuonPlus_beforeRad_found && genMuonPlus_afterRad_found);
    bool genMuonMinus_found = (genMuonMinus_beforeRad_found && genMuonMinus_afterRad_found);

    if ( genTauPlus && genMuonPlus_found && genTauMinus && genMuonMinus_found ) {
      for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator muonRadCorrUncertaintyPlotEntry = muonRadCorrUncertaintyPlotEntries_beforeRad_.begin();
	    muonRadCorrUncertaintyPlotEntry != muonRadCorrUncertaintyPlotEntries_beforeRad_.end(); ++muonRadCorrUncertaintyPlotEntry ) {
	(*muonRadCorrUncertaintyPlotEntry)->fillHistograms(numJetsCorrPtGt30, genMuonPlusP4_beforeRad, genMuonMinusP4_beforeRad, evtWeight, muonRadCorrWeight, muonRadCorrWeightUp, muonRadCorrWeightDown);
      }
      for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator muonRadCorrUncertaintyPlotEntry = muonRadCorrUncertaintyPlotEntries_afterRad_.begin();
	    muonRadCorrUncertaintyPlotEntry != muonRadCorrUncertaintyPlotEntries_afterRad_.end(); ++muonRadCorrUncertaintyPlotEntry ) {
	(*muonRadCorrUncertaintyPlotEntry)->fillHistograms(numJetsCorrPtGt30, genMuonPlusP4_afterRad, genMuonMinusP4_afterRad, evtWeight, muonRadCorrWeight, muonRadCorrWeightUp, muonRadCorrWeightDown);
      }
      for ( std::vector<plotEntryTypeMuonRadCorrUncertainty*>::iterator muonRadCorrUncertaintyPlotEntry = muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_.begin();
	    muonRadCorrUncertaintyPlotEntry != muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_.end(); ++muonRadCorrUncertaintyPlotEntry ) {
	(*muonRadCorrUncertaintyPlotEntry)->fillHistograms(numJetsCorrPtGt30, genTauPlus->p4(), genTauMinus->p4(), evtWeight, muonRadCorrWeight, muonRadCorrWeightUp, muonRadCorrWeightDown);
      }
    } else {
      if ( muonRadCorrUncertainty_numWarnings_ < muonRadCorrUncertainty_maxWarnings_ ) {
	edm::LogWarning ("<MCEmbeddingValidationAnalyzer::analyze>")
	  << " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
	  << " Failed to match muons before and after radiation to embedded tau leptons !!" << std::endl;
	std::cout << "genTauPlus: ";
	if ( genTauPlus ) std::cout << "Pt = " << genTauPlus->pt() << ", eta = " << genTauPlus->eta() << ", phi = " << genTauPlus->phi() << std::endl;
	else std::cout << "NA" << std::endl;
	std::cout << "genMuonPlus (before Rad.): ";
	if ( genMuonPlus_found ) std::cout << "Pt = " << genMuonPlusP4_beforeRad.pt() << ", eta = " << genMuonPlusP4_beforeRad.eta() << ", phi = " << genMuonPlusP4_beforeRad.phi() << std::endl;
	else std::cout << "NA" << std::endl;
	std::cout << "genTauMinus: ";
	if ( genTauMinus ) std::cout << "Pt = " << genTauMinus->pt() << ", eta = " << genTauMinus->eta() << ", phi = " << genTauMinus->phi() << std::endl;
	else std::cout << "NA" << std::endl;
	std::cout << "genMuonMinus (before Rad.): ";
	if ( genMuonMinus_found ) std::cout << "Pt = " << genMuonMinusP4_beforeRad.pt() << ", eta = " << genMuonMinusP4_beforeRad.eta() << ", phi = " << genMuonMinusP4_beforeRad.phi() << std::endl;
	else std::cout << "NA" << std::endl;
	++muonRadCorrUncertainty_numWarnings_;
      }
    }
  }

  edm::Handle<l1extra::L1EtMissParticleCollection> l1METs;
  evt.getByLabel(srcL1ETM_, l1METs);
  const reco::Candidate::LorentzVector& l1MEtP4 = l1METs->front().p4();
  //std::cout << "L1MEt: Pt = " << l1MEtP4.pt() << ", phi = " << l1MEtP4.phi() << " (Et = " << l1METs->front().etMiss() << ", Px = " << l1MEtP4.px() << ", Py = " << l1MEtP4.py() << ")" << std::endl;
  //if ( l1MEtP4.pt() > 75. ) std::cout << "--> CHECK !!" << std::endl;
  typedef edm::View<reco::MET> METView;
  edm::Handle<METView> recCaloMETs;
  evt.getByLabel(srcRecCaloMEt_, recCaloMETs);
  const reco::Candidate::LorentzVector& recCaloMEtP4 = recCaloMETs->front().p4();
  //std::cout << "recCaloMEt: Pt = " << recCaloMEtP4.pt() << ", phi = " << recCaloMEtP4.phi() << " (Px = " << recCaloMEtP4.px() << ", Py = " << recCaloMEtP4.py() << ")" << std::endl;
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
      (*l1ETMplotEntry)->fillHistograms(genTauDecayMode_ref, l1MEtP4, genCaloMEtP4, recCaloMEtP4, genDiTau->p4(), muonRadCorrWeight*evtWeight);
    }
  }

  fillHistograms(electronDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(electronDistributionsExtra_, numJetsCorrPtGt30, evt, es, muonRadCorrWeight*evtWeight);
  fillHistograms(electronEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(gsfElectronEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(electronL1TriggerEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(muonDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(muonEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(muonL1TriggerEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(tauDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(tauDistributionsExtra_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(tauEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(l1ElectronDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(l1MuonDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(l1TauDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(l1CentralJetDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(l1ForwardJetDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(metDistributions_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);
  fillHistograms(metL1TriggerEfficiencies_, numJetsCorrPtGt30, evt, muonRadCorrWeight*evtWeight);

  // CV: Check for presence in the embedded event of high Pt tracks, charged PFCandidates and muons 
  //     reconstructed near the direction of the replaced muons
  //    (indicating that maybe not all muon signals were removed in the embedding)
  std::vector<const reco::Candidate*> replacedMuons;
  if ( replacedMuonPlus  ) replacedMuons.push_back(replacedMuonPlus);
  if ( replacedMuonMinus ) replacedMuons.push_back(replacedMuonMinus);
  for ( std::vector<const reco::Candidate*>::const_iterator replacedMuon = replacedMuons.begin();
	replacedMuon != replacedMuons.end(); ++replacedMuon ) {
    for ( reco::TrackCollection::const_iterator track = tracks->begin();
	  track != tracks->end(); ++track ) {
      if ( track->pt() > 10. ) {
	double dR = deltaR(track->eta(), track->phi(), (*replacedMuon)->eta(), (*replacedMuon)->phi());
	if ( dR < 0.1 ) {
	  edm::LogWarning("MCEmbeddingValidationAnalyzer") 
	    << " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
	    << " Found track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << ", charge = " << track->charge() << " in direction of"
	    << " a replaced muon: Pt = " << (*replacedMuon)->pt()<< ", eta = " << (*replacedMuon)->eta() << ", phi = " << (*replacedMuon)->phi() << ", charge = " << (*replacedMuon)->charge()
	    << " (dR = " << dR << "). This may point to a problem in removing all muon signals in the Embedding." << std::endl;
	  histogramWarning_recTrackNearReplacedMuon_->Fill(track->charge()*(*replacedMuon)->charge(), muonRadCorrWeight*evtWeight);
	}
      }
    }
    for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	  pfCandidate != pfCandidates->end(); ++pfCandidate ) {
      if ( pfCandidate->pt() > 10. && TMath::Abs(pfCandidate->charge()) > 0.5 ) {
	double dR = deltaR(pfCandidate->eta(), pfCandidate->phi(), (*replacedMuon)->eta(), (*replacedMuon)->phi());
	if ( dR < 0.1 ) {
	  edm::LogWarning("MCEmbeddingValidationAnalyzer") 
	    << " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
	    << " Found charged PFCandidate: Pt = " << pfCandidate->pt() << ", eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() << ", charge = " << pfCandidate->charge() << " in direction of"
	    << " a replaced muon: Pt = " << (*replacedMuon)->pt()<< ", eta = " << (*replacedMuon)->eta() << ", phi = " << (*replacedMuon)->phi() << ", charge = " << (*replacedMuon)->charge()
	    << " (dR = " << dR << "). This may point to a problem in removing all muon signals in the Embedding." << std::endl;
	  histogramWarning_recPFCandNearReplacedMuon_->Fill(pfCandidate->charge()*(*replacedMuon)->charge(), muonRadCorrWeight*evtWeight);
	}
      }
    }
    for ( reco::MuonCollection::const_iterator muon = muons->begin();
	  muon != muons->end(); ++muon ) {
      if ( muon->pt() > 10. ) {
	double dR = deltaR(muon->eta(), muon->phi(), (*replacedMuon)->eta(), (*replacedMuon)->phi());
	if ( dR < 0.1 ) {
	  edm::LogWarning("MCEmbeddingValidationAnalyzer") 
	    << " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
	    << " Found track: Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << ", charge = " << muon->charge() << " in direction of"
	    << " a replaced muon: Pt = " << (*replacedMuon)->pt()<< ", eta = " << (*replacedMuon)->eta() << ", phi = " << (*replacedMuon)->phi() << ", charge = " << (*replacedMuon)->charge()
	    << " (dR = " << dR << "). This may point to a problem in removing all muon signals in the Embedding." << std::endl;
	  histogramWarning_recMuonNearReplacedMuon_->Fill(muon->charge()*(*replacedMuon)->charge(), muonRadCorrWeight*evtWeight);
	}
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCEmbeddingValidationAnalyzer);
