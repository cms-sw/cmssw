#ifndef TauAnalysis_MCEmbeddingTools_MCEmbeddingValidationAnalyzer_h
#define TauAnalysis_MCEmbeddingTools_MCEmbeddingValidationAnalyzer_h

/** \class MCEmbeddingValidationAnalyzer
 *
 * Compare Ztautau events produced via MCEmbedding 
 * to Ztautau events produced via direct Monte Carlo production in terms of:
 *   o kinematic distributions of electrons, muons, taus and MET
 *   o electron, muon and tau reconstruction and identification efficiencies
 *   o MET resolution
 *   o trigger efficiencies
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MCEmbeddingValidationAnalyzer.h,v 1.1 2012/11/12 08:02:35 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include <TString.h>

#include <string>
#include <vector>

class MCEmbeddingValidationAnalyzer : public edm::EDAnalyzer 
{
 public:
  explicit MCEmbeddingValidationAnalyzer(const edm::ParameterSet&);
  ~MCEmbeddingValidationAnalyzer();
    
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:
  std::string dqmDirectory_full(const std::string& dqmSubDirectory)
  {
    TString dqmDirectory_full = dqmDirectory_.data();
    if ( !dqmDirectory_full.EndsWith("/") ) dqmDirectory_full.Append("/");
    dqmDirectory_full.Append(dqmSubDirectory);
    return dqmDirectory_full.Data();
  }
  
  edm::InputTag srcMuons_;
  edm::InputTag srcTracks_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcWeights_;

  std::string dqmDirectory_;

  MonitorElement* histogramNumTracksPtGt5_;
  MonitorElement* histogramNumTracksPtGt10_;
  MonitorElement* histogramNumTracksPtGt20_;
  MonitorElement* histogramNumTracksPtGt30_;
  MonitorElement* histogramNumTracksPtGt40_;

  MonitorElement* histogramNumGlobalMuons_;
  MonitorElement* histogramNumStandAloneMuons_;
  MonitorElement* histogramNumPFMuons_;
  
  template <typename T>
  struct leptonDistributionT
  {
    leptonDistributionT(const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory)
      : srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
	dRmatch_(dRmatch),
	dqmDirectory_(dqmDirectory)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<T>(cutRec);
    }
    ~leptonDistributionT() 
    {
      delete cutGen_;
      delete cutRec_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramGenLeptonPt_ = dqmStore.book1D("genLeptonPt", "genLeptonPt", 250, 0., 250.);
      histogramGenLeptonEta_ = dqmStore.book1D("genLeptonEta", "genLeptonEta", 198, -9.9, +9.9);
      histogramGenLeptonPhi_ = dqmStore.book1D("genLeptonPhi", "genLeptonPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecLeptonPt_ = dqmStore.book1D("recLeptonPt", "recLeptonPt", 250, 0., 250.);
      histogramRecLeptonEta_ = dqmStore.book1D("recLeptonEta", "recLeptonEta", 198, -9.9, +9.9);
      histogramRecLeptonPhi_ = dqmStore.book1D("recLeptonPhi", "recLeptonPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMinusGenLeptonPt_ = dqmStore.book1D("recMinusGenLeptonPt", "recMinusGenLeptonPt", 200, -100., +100.);
      histogramRecMinusGenLeptonEta_ = dqmStore.book1D("recMinusGenLeptonEta", "recMinusGenLeptonEta", 100, -0.5, +0.5);
      histogramRecMinusGenLeptonPhi_ = dqmStore.book1D("recMinusGenLeptonPhi", "recMinusGenLeptonPhi", 100, -0.5, +0.5);
    }
    void fillHistograms(const edm::Event& evt, double evtWeight)
    {
      typedef edm::View<reco::Candidate> CandidateView;
      edm::Handle<CandidateView> genLeptons;
      evt.getByLabel(srcGen_, genLeptons);
      typedef std::vector<T> recLeptonCollection;
      edm::Handle<recLeptonCollection> recLeptons;
      evt.getByLabel(srcRec_, recLeptons);
      for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	    genLepton != genLeptons->end(); ++genLepton ) {
	if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;
	for ( typename recLeptonCollection::const_iterator recLepton = recLeptons->begin();
	      recLepton != recLeptons->end(); ++recLepton ) {
	  if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	  double dR = deltaR(genLepton->p4(), recLepton->p4());
	  if ( dR < dRmatch_ ) {
	    histogramGenLeptonPt_->Fill(genLepton->pt(), evtWeight);
	    histogramGenLeptonEta_->Fill(genLepton->eta(), evtWeight);
	    histogramGenLeptonPhi_->Fill(genLepton->phi(), evtWeight);
	    histogramRecLeptonPt_->Fill(recLepton->pt(), evtWeight);
	    histogramRecLeptonEta_->Fill(recLepton->eta(), evtWeight);
	    histogramRecLeptonPhi_->Fill(recLepton->phi(), evtWeight);
	    histogramRecMinusGenLeptonPt_->Fill(recLepton->pt() - genLepton->pt(), evtWeight);
	    histogramRecMinusGenLeptonEta_->Fill(recLepton->eta() - genLepton->eta(), evtWeight);
	    histogramRecMinusGenLeptonPhi_->Fill(recLepton->phi() - genLepton->phi(), evtWeight);
	  }	     
	}
      }
    }
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<T>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramGenLeptonPt_;
    MonitorElement* histogramGenLeptonEta_;
    MonitorElement* histogramGenLeptonPhi_;
    MonitorElement* histogramRecLeptonPt_;
    MonitorElement* histogramRecLeptonEta_;
    MonitorElement* histogramRecLeptonPhi_;
    MonitorElement* histogramRecMinusGenLeptonPt_;
    MonitorElement* histogramRecMinusGenLeptonEta_;
    MonitorElement* histogramRecMinusGenLeptonPhi_;
  };

  template <typename T>
  void setupLeptonDistribution(const edm::ParameterSet&, const std::string&, std::vector<leptonDistributionT<T>*>&);

  std::vector<leptonDistributionT<pat::Electron>*> electronDistributions_;
  std::vector<leptonDistributionT<pat::Muon>*> muonDistributions_;
  std::vector<leptonDistributionT<pat::Tau>*> tauDistributions_;

  template <typename T>
  struct leptonEfficiencyT
  {
    leptonEfficiencyT(const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory)
      : srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
	dRmatch_(dRmatch),
	dqmDirectory_(dqmDirectory)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<T>(cutRec);
    }
    ~leptonEfficiencyT() 
    {
      delete cutGen_;
      delete cutRec_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorEta_ = dqmStore.book1D("numeratorEta", "numeratorEta", 198, -9.9, +9.9);
      histogramDenominatorEta_ = dqmStore.book1D("denominatorEta", "denominatorEta", 198, -9.9, +9.9);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(const edm::Event& evt, double evtWeight)
    {
      typedef edm::View<reco::Candidate> CandidateView;
      edm::Handle<CandidateView> genLeptons;
      evt.getByLabel(srcGen_, genLeptons);
      typedef std::vector<T> recLeptonCollection;
      edm::Handle<recLeptonCollection> recLeptons;
      evt.getByLabel(srcRec_, recLeptons);
      for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	    genLepton != genLeptons->end(); ++genLepton ) {
	if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;	
	bool isMatched = false;
	for ( typename recLeptonCollection::const_iterator recLepton = recLeptons->begin();
	      recLepton != recLeptons->end(); ++recLepton ) {
	  if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	  double dR = deltaR(genLepton->p4(), recLepton->p4());
	  if ( dR < dRmatch_ ) isMatched = true;
	}
	histogramDenominatorPt_->Fill(genLepton->pt(), evtWeight);
	histogramDenominatorEta_->Fill(genLepton->eta(), evtWeight);
	histogramDenominatorPhi_->Fill(genLepton->phi(), evtWeight);
	if ( isMatched ) {
	  histogramNumeratorPt_->Fill(genLepton->pt(), evtWeight);
	  histogramNumeratorEta_->Fill(genLepton->eta(), evtWeight);
	  histogramNumeratorPhi_->Fill(genLepton->phi(), evtWeight);
	}
      }
    }
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<T>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorEta_;
    MonitorElement* histogramDenominatorEta_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };

  template <typename T>
  void setupLeptonEfficiency(const edm::ParameterSet&, const std::string&, std::vector<leptonEfficiencyT<T>*>&);

  std::vector<leptonEfficiencyT<pat::Electron>*> electronEfficiencies_;
  std::vector<leptonEfficiencyT<pat::Muon>*> muonEfficiencies_;
  std::vector<leptonEfficiencyT<pat::Tau>*> tauEfficiencies_;

  template <typename T1, typename T2>
  struct leptonL1TriggerEfficiencyT1T2
  {
    leptonL1TriggerEfficiencyT1T2(const edm::InputTag& srcRef, const std::string& cutRef, const edm::InputTag& srcL1, const std::string& cutL1, double dRmatch, const std::string& dqmDirectory)
      : srcRef_(srcRef),
	cutRef_(0),
	srcL1_(srcL1),
	cutL1_(0),
	dRmatch_(dRmatch),
	dqmDirectory_(dqmDirectory)
    {
      if ( cutRef != "" ) cutRef_ = new StringCutObjectSelector<T1>(cutRef);
      if ( cutL1  != "" ) cutL1_  = new StringCutObjectSelector<T2>(cutL1);
    }
    ~leptonL1TriggerEfficiencyT1T2() 
    {
      delete cutRef_;
      delete cutL1_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorEta_ = dqmStore.book1D("numeratorEta", "numeratorEta", 198, -9.9, +9.9);
      histogramDenominatorEta_ = dqmStore.book1D("denominatorEta", "denominatorEta", 198, -9.9, +9.9);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(const edm::Event& evt, double evtWeight)
    {
      typedef std::vector<T1> refLeptonCollection;
      edm::Handle<refLeptonCollection> refLeptons;
      evt.getByLabel(srcRef_, refLeptons);
      typedef std::vector<T2> l1LeptonCollection;
      edm::Handle<l1LeptonCollection> l1Leptons;
      evt.getByLabel(srcL1_, l1Leptons);
      for ( typename refLeptonCollection::const_iterator refLepton = refLeptons->begin();
	    refLepton != refLeptons->end(); ++refLepton ) {
	if ( cutRef_ && !(*cutRef_)(*refLepton) ) continue;
	bool isMatched = false;
	for ( typename l1LeptonCollection::const_iterator l1Lepton = l1Leptons->begin();
	      l1Lepton != l1Leptons->end(); ++l1Lepton ) {
	  if ( cutL1_ && !(*cutL1_)(*l1Lepton) ) continue;
	  double dR = deltaR(refLepton->p4(), l1Lepton->p4());
	  if ( dR < dRmatch_ ) isMatched = true;
	}
	histogramDenominatorPt_->Fill(refLepton->pt(), evtWeight);
	histogramDenominatorEta_->Fill(refLepton->eta(), evtWeight);
	histogramDenominatorPhi_->Fill(refLepton->phi(), evtWeight);
	if ( isMatched ) {
	  histogramNumeratorPt_->Fill(refLepton->pt(), evtWeight);
	  histogramNumeratorEta_->Fill(refLepton->eta(), evtWeight);
	  histogramNumeratorPhi_->Fill(refLepton->phi(), evtWeight);
	}
      }
    }
    edm::InputTag srcRef_;
    StringCutObjectSelector<T1>* cutRef_;
    edm::InputTag srcL1_;
    StringCutObjectSelector<T2>* cutL1_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorEta_;
    MonitorElement* histogramDenominatorEta_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };

  template <typename T1, typename T2>
  void setupLeptonL1TriggerEfficiency(const edm::ParameterSet&, const std::string&, std::vector<leptonL1TriggerEfficiencyT1T2<T1,T2>*>&);

  std::vector<leptonL1TriggerEfficiencyT1T2<pat::Electron, l1extra::L1EmParticle>*> electronL1TriggerEfficiencies_;
  std::vector<leptonL1TriggerEfficiencyT1T2<pat::Muon, l1extra::L1MuonParticle>*> muonL1TriggerEfficiencies_;

  struct metDistributionType
  {
    metDistributionType(const edm::InputTag& srcGen, const edm::InputTag& srcRec, const edm::InputTag& srcGenZs, const std::string& dqmDirectory)
      : srcGen_(srcGen),
	srcRec_(srcRec),
	srcGenZs_(srcGenZs),
	dqmDirectory_(dqmDirectory)
    {}
    ~metDistributionType() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramGenMEtPt_  = dqmStore.book1D("genMEtPt", "genMEtPt", 250, 0., 250.);
      histogramGenMEtPhi_ = dqmStore.book1D("genMEtPhi", "genMEtPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMEtPt_  = dqmStore.book1D("recMEtPt", "recMEtPt", 250, 0., 250.);
      histogramRecMEtPhi_ = dqmStore.book1D("recMEtPhi", "recMEtPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMinusGenMEtParlZ_ = dqmStore.book1D("histogramRecMinusGenMEtParlZ", "histogramRecMinusGenMEtParlZ", 200, -100., +100.);
      histogramRecMinusGenMEtPerpZ_ = dqmStore.book1D("histogramRecMinusGenMEtPerpZ", "histogramRecMinusGenMEtPerpZ", 100, 0., 100.);
    }
    void fillHistograms(const edm::Event& evt, double evtWeight)
    {
      typedef edm::View<reco::MET> METView;
      edm::Handle<METView> genMETs;
      evt.getByLabel(srcGen_, genMETs);
      const reco::Candidate::LorentzVector& genMEtP4 = genMETs->front().p4();
      edm::Handle<METView> recMETs;
      evt.getByLabel(srcRec_, recMETs);
      const reco::Candidate::LorentzVector& recMEtP4 = recMETs->front().p4();
      typedef edm::View<reco::Candidate> CandidateView;
      edm::Handle<CandidateView> genZs;
      evt.getByLabel(srcGenZs_, genZs);
      if ( !(genZs->size() >= 1) ) return;
      const reco::Candidate::LorentzVector& genZp4 = genZs->front().p4();
      histogramGenMEtPt_->Fill(genMEtP4.pt(), evtWeight);
      histogramGenMEtPhi_->Fill(genMEtP4.phi(), evtWeight);
      histogramRecMEtPt_->Fill(recMEtP4.pt(), evtWeight);
      histogramRecMEtPhi_->Fill(recMEtP4.phi(), evtWeight);
      if ( genZp4.pt() > 0. ) {
	double qX = genZp4.px();
	double qY = genZp4.py();
	double qT = TMath::Sqrt(qX*qX + qY*qY);
	double dX = recMEtP4.px() - genMEtP4.px();
	double dY = recMEtP4.py() - genMEtP4.py();
	double dParl = (dX*qX + dY*qY)/qT;
	double dPerp = (dX*qY - dY*qX)/qT;
	histogramRecMinusGenMEtParlZ_->Fill(dParl, evtWeight);
	histogramRecMinusGenMEtPerpZ_->Fill(TMath::Abs(dPerp), evtWeight);
      }
    }
    edm::InputTag srcGen_;
    edm::InputTag srcRec_;
    edm::InputTag srcGenZs_;
    std::string dqmDirectory_;
    MonitorElement* histogramGenMEtPt_;
    MonitorElement* histogramGenMEtPhi_;
    MonitorElement* histogramRecMEtPt_;
    MonitorElement* histogramRecMEtPhi_;
    MonitorElement* histogramRecMinusGenMEtParlZ_;
    MonitorElement* histogramRecMinusGenMEtPerpZ_;
  };

  void setupMEtDistribution(const edm::ParameterSet&, const std::string&, std::vector<metDistributionType*>&);

  std::vector<metDistributionType*> metDistributions_;

  struct metL1TriggerEfficiencyType
  {
    metL1TriggerEfficiencyType(const edm::InputTag& srcRef, const edm::InputTag& srcL1, double cutL1Pt, const std::string& dqmDirectory)
      : srcRef_(srcRef),
	srcL1_(srcL1),
	cutL1Pt_(cutL1Pt),
	dqmDirectory_(dqmDirectory)
    {}
    ~metL1TriggerEfficiencyType() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(const edm::Event& evt, double evtWeight)
    {
      typedef edm::View<reco::MET> METView;
      edm::Handle<METView> refMETs;
      evt.getByLabel(srcRef_, refMETs);
      const reco::Candidate::LorentzVector& refMEtP4 = refMETs->front().p4();
      edm::Handle<l1extra::L1EtMissParticleCollection> l1METs;
      evt.getByLabel(srcL1_, l1METs);
      double l1MEt = l1METs->begin()->etMiss();
      histogramDenominatorPt_->Fill(refMEtP4.pt(), evtWeight);
      histogramDenominatorPhi_->Fill(refMEtP4.phi(), evtWeight);
      if ( l1MEt > cutL1Pt_ ) {
	histogramNumeratorPt_->Fill(refMEtP4.pt(), evtWeight);
	histogramNumeratorPhi_->Fill(refMEtP4.phi(), evtWeight);
      }
    }
    edm::InputTag srcRef_;
    edm::InputTag srcL1_;
    double cutL1Pt_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };
  
  void setupMEtL1TriggerEfficiency(const edm::ParameterSet&, const std::string&, std::vector<metL1TriggerEfficiencyType*>&);

  std::vector<metL1TriggerEfficiencyType*> metL1TriggerEfficiencies_;

  template <typename T>
  void cleanCollection(std::vector<T*> collection)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      delete (*object);
    }
  }

  template <typename T>
  void bookHistograms(std::vector<T*> collection, DQMStore& dqmStore)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      (*object)->bookHistograms(dqmStore);
    }
  } 

  template <typename T>
  void fillHistograms(std::vector<T*> collection, const edm::Event& evt, double evtWeight)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      (*object)->fillHistograms(evt, evtWeight);
    }
  } 

  
};

#endif
