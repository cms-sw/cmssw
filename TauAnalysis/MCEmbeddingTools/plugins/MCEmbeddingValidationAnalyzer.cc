#include "TauAnalysis/MCEmbeddingTools/plugins/MCEmbeddingValidationAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

MCEmbeddingValidationAnalyzer::MCEmbeddingValidationAnalyzer(const edm::ParameterSet& cfg)
  : srcReplacedMuons_(cfg.getParameter<edm::InputTag>("srcReplacedMuons")),
    srcRecMuons_(cfg.getParameter<edm::InputTag>("srcRecMuons")),
    srcRecTracks_(cfg.getParameter<edm::InputTag>("srcRecTracks")),
    srcRecPFCandidates_(cfg.getParameter<edm::InputTag>("srcRecPFCandidates")),
    srcRecVertex_(cfg.getParameter<edm::InputTag>("srcRecVertex")),
    srcGenDiTaus_(cfg.getParameter<edm::InputTag>("srcGenDiTaus")),
    srcGenLeg1_(cfg.getParameter<edm::InputTag>("srcGenLeg1")),
    srcRecLeg1_(cfg.getParameter<edm::InputTag>("srcRecLeg1")),
    srcGenLeg2_(cfg.getParameter<edm::InputTag>("srcGenLeg2")),
    srcRecLeg2_(cfg.getParameter<edm::InputTag>("srcRecLeg2")),
    srcWeights_(cfg.getParameter<vInputTag>("srcWeights")),
    srcGenFilterInfo_(cfg.getParameter<edm::InputTag>("srcGenFilterInfo")),
    dqmDirectory_(cfg.getParameter<std::string>("dqmDirectory")),
    replacedMuonPtThresholdHigh_(cfg.getParameter<double>("replacedMuonPtThresholdHigh")),
    replacedMuonPtThresholdLow_(cfg.getParameter<double>("replacedMuonPtThresholdLow"))
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
  setupTauDistributionExtra(cfg, "tauExtraDistributions", tauDistributionsExtra_);
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
  cleanCollection(tauDistributionsExtra_);
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

void MCEmbeddingValidationAnalyzer::setupTauDistributionExtra(const edm::ParameterSet& cfg, const std::string& keyword, std::vector<tauDistributionExtra*>& tauDistributionsExtra)
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
      tauDistributionExtra* tauDistribution = new tauDistributionExtra(srcGen, cutGen, srcRec, cutRec, dRmatch, dqmDirectory);
      tauDistributionsExtra.push_back(tauDistribution);
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
  histogramGenFilterEfficiency_     = dqmStore.book1D("genFilterEfficiency",    "genFilterEfficiency",    102,     -0.01,         1.01);

  histogramRotationAngleMatrix_     = dqmStore.book2D("rfRotationAngleMatrix",  "rfRotationAngleMatrix",    2,     -0.5,          1.5, 2, -0.5, 1.5);

  histogramNumTracksPtGt5_          = dqmStore.book1D("numTracksPtGt5",         "numTracksPtGt5",          50,     -0.5,         49.5);
  histogramNumTracksPtGt10_         = dqmStore.book1D("numTracksPtGt10",        "numTracksPtGt10",         50,     -0.5,         49.5);
  histogramNumTracksPtGt20_         = dqmStore.book1D("numTracksPtGt20",        "numTracksPtGt20",         50,     -0.5,         49.5);
  histogramNumTracksPtGt30_         = dqmStore.book1D("numTracksPtGt30",        "numTracksPtGt30",         50,     -0.5,         49.5);
  histogramNumTracksPtGt40_         = dqmStore.book1D("numTracksPtGt40",        "numTracksPtGt40",         50,     -0.5,         49.5);
      
  histogramNumGlobalMuons_          = dqmStore.book1D("numGlobalMuons",         "numGlobalMuons",          20,     -0.5,         19.5);
  histogramNumStandAloneMuons_      = dqmStore.book1D("numStandAloneMuons",     "numStandAloneMuons",      20,     -0.5,         19.5);
  histogramNumPFMuons_              = dqmStore.book1D("numPFMuons",             "numPFMuons",              20,     -0.5,         19.5);

  histogramNumChargedPFCandsPtGt5_  = dqmStore.book1D("numChargedPFCandsPtGt5", "numChargedPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt10_ = dqmStore.book1D("numChargedPFCandsPtGt5", "numChargedPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt20_ = dqmStore.book1D("numChargedPFCandsPtGt5", "numChargedPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt30_ = dqmStore.book1D("numChargedPFCandsPtGt5", "numChargedPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumChargedPFCandsPtGt40_ = dqmStore.book1D("numChargedPFCandsPtGt5", "numChargedPFCandsPtGt5",  50,     -0.5,         49.5);

  histogramNumNeutralPFCandsPtGt5_  = dqmStore.book1D("numNeutralPFCandsPtGt5", "numNeutralPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt10_ = dqmStore.book1D("numNeutralPFCandsPtGt5", "numNeutralPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt20_ = dqmStore.book1D("numNeutralPFCandsPtGt5", "numNeutralPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt30_ = dqmStore.book1D("numNeutralPFCandsPtGt5", "numNeutralPFCandsPtGt5",  50,     -0.5,         49.5);
  histogramNumNeutralPFCandsPtGt40_ = dqmStore.book1D("numNeutralPFCandsPtGt5", "numNeutralPFCandsPtGt5",  50,     -0.5,         49.5);
    
  histogramRecVertexX_              = dqmStore.book1D("recVertexX",             "recVertexX",            2000,     -1.,          +1.);
  histogramRecVertexY_              = dqmStore.book1D("recVertexY",             "recVertexY",            2000,     -1.,          +1.);
  histogramRecVertexZ_              = dqmStore.book1D("recVertexZ",             "recVertexZ",             500,    -25.,         +25.);
  
  histogramGenDiTauPt_              = dqmStore.book1D("genDiTauPt",             "genDiTauPt",             250,      0.,         250.);
  histogramGenDiTauEta_             = dqmStore.book1D("genDiTauEta",            "genDiTauEta",            198,     -9.9,         +9.9);
  histogramGenDiTauPhi_             = dqmStore.book1D("genDiTauPhi",            "genDiTauPhi",             72, -TMath::Pi(), +TMath::Pi());
  histogramGenDiTauMass_            = dqmStore.book1D("genDiTauMass",           "genDiTauMass",           500,      0.,         500.);

  histogramGenVisDiTauPt_           = dqmStore.book1D("genVisDiTauPt",          "genVisDiTauPt",          250,      0.,         250.);
  histogramGenVisDiTauEta_          = dqmStore.book1D("genVisDiTauEta",         "genVisDiTauEta",         198,     -9.9,         +9.9);
  histogramGenVisDiTauPhi_          = dqmStore.book1D("genVisDiTauPhi",         "genVisDiTauPhi",          72, -TMath::Pi(), +TMath::Pi());
  histogramGenVisDiTauMass_         = dqmStore.book1D("genVisDiTauMass",        "genVisDiTauMass",        500,      0.,         500.);

  histogramRecVisDiTauPt_           = dqmStore.book1D("recVisDiTauPt",          "recVisDiTauPt",          250,      0.,         250.);
  histogramRecVisDiTauEta_          = dqmStore.book1D("recVisDiTauEta",         "recVisDiTauEta",         198,     -9.9,         +9.9);
  histogramRecVisDiTauPhi_          = dqmStore.book1D("recVisDiTauPhi",         "recVisDiTauPhi",          72, -TMath::Pi(), +TMath::Pi());
  histogramRecVisDiTauMass_         = dqmStore.book1D("recVisDiTauMass",        "recVisDiTauMass",        500,      0.,         500.);

  histogramGenLeg1Pt_               = dqmStore.book1D("genLeg1Pt",              "genLeg1Pt",              250,      0.,         250.);
  histogramGenLeg1Eta_              = dqmStore.book1D("genLeg1Eta",             "genLeg1Eta",             198,     -9.9,         +9.9);
  histogramGenLeg1Phi_              = dqmStore.book1D("genLeg1Phi",             "genLeg1Phi",              72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg1X_                = dqmStore.book1D("genLeg1X",               "genLeg1X",               102,     -0.01,         1.01);
  histogramRecLeg1X_                = dqmStore.book1D("recLeg1X",               "recLeg1X",               102,     -0.01,         1.01);
  histogramGenLeg2Pt_               = dqmStore.book1D("genLeg2Pt",              "genLeg2Pt",              250,      0.,         250.);
  histogramGenLeg2Eta_              = dqmStore.book1D("genLeg2Eta",             "genLeg2Eta",             198,     -9.9,         +9.9);
  histogramGenLeg2Phi_              = dqmStore.book1D("genLeg2Phi",             "genLeg2Phi",              72, -TMath::Pi(), +TMath::Pi());
  histogramGenLeg2X_                = dqmStore.book1D("genLeg2X",               "genLeg2X",               102,     -0.01,         1.01);
  histogramRecLeg2X_                = dqmStore.book1D("recLeg2X",               "recLeg2X",               102,     -0.01,         1.01);

  bookHistograms(electronDistributions_, dqmStore);
  bookHistograms(electronEfficiencies_, dqmStore);
  bookHistograms(electronL1TriggerEfficiencies_, dqmStore);
  bookHistograms(muonDistributions_, dqmStore);
  bookHistograms(muonEfficiencies_, dqmStore);
  bookHistograms(muonL1TriggerEfficiencies_, dqmStore);
  bookHistograms(tauDistributions_, dqmStore);
  bookHistograms(tauDistributionsExtra_, dqmStore);
  bookHistograms(tauEfficiencies_, dqmStore);
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
	    //std::cout << "leg2(" << srcLeg2.label() << "): Pt = " << visDecayProduct2->pt() << ", eta = " << visDecayProduct2->eta() << ", phi = " << visDecayProduct2->phi() << std::endl;
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
    const reco::Candidate* genLeg1 = genDiTau_composite->daughter(0);
    const reco::Candidate* genLeg2 = genDiTau_composite->daughter(1);
    if ( !(genLeg1 && genLeg2) ) continue;
    if ( (genLeg1->pt() > replacedMuonPtThresholdHigh_ && genLeg2->pt() > replacedMuonPtThresholdLow_ ) ||
	 (genLeg1->pt() > replacedMuonPtThresholdLow_  && genLeg2->pt() > replacedMuonPtThresholdHigh_) ) {
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
  
  fillHistograms(electronDistributions_, evt, evtWeight);
  fillHistograms(electronEfficiencies_, evt, evtWeight);
  fillHistograms(electronL1TriggerEfficiencies_, evt, evtWeight);
  fillHistograms(muonDistributions_, evt, evtWeight);
  fillHistograms(muonEfficiencies_, evt, evtWeight);
  fillHistograms(muonL1TriggerEfficiencies_, evt, evtWeight);
  fillHistograms(tauDistributions_, evt, evtWeight);
  fillHistograms(tauDistributionsExtra_, evt, evtWeight);
  fillHistograms(tauEfficiencies_, evt, evtWeight);
  fillHistograms(metDistributions_, evt, evtWeight);
  fillHistograms(metL1TriggerEfficiencies_, evt, evtWeight);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCEmbeddingValidationAnalyzer);
