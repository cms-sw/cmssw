#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"

#include <vector>
#include <string>
#include <iostream>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TH1F.h>

//
// class declaration
//
class SimpleCaloHit2;
class SimpleCluster;


class EcalEBTrigPrimAnalyzer : public edm::one::EDAnalyzer<> {
 public:
  explicit EcalEBTrigPrimAnalyzer(const edm::ParameterSet&);
  ~EcalEBTrigPrimAnalyzer();
  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();
  
  std::vector<SimpleCluster> makeCluster  ( std::vector<SimpleCaloHit2> & hitCollection, int dEta, int dPhi );
  float etaTransformation (float eta, float zPV);  
  float normalizedPhi(float phi);

private:

  std::string outputFileName_;

  std::unique_ptr<PhotonMCTruthFinder> thePhotonMCTruthFinder_;

  int nEvents_;
  int nTotTP_;
  int nTotCluTP_;
  int nTotCluRH_;
  void init(const edm::EventSetup&);

  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I * ecal_et_[2];
  TH1I * ecal_tt_[2];
  TH1I * ecal_fgvb_[2];
  TH1I *histEndc,*histBar;
  TFile *histfile_;
  TH1F *hAllTPperEvt_;
  TH1F *hTPperEvt_;
  TH2F *hTPvsRechit_;
  TH1F *hTPoverRechit_;
  TH1F *hDeltaEt_;
  TH1F *hAllRechitEt_;
  TH1F *hRechitEt_;
  TH1F *hTPEt_;
  TH1F *hRatioEt_;
  TH2F *hTPvsTow_ieta_;
  TH2F *hTPvsTow_eta_;
  TH2F *hTP_iphiVsieta_;
  TH2F *hRH_iphiVsieta_;
  TH2F *hTP_iphiVsieta_fullrange_;
  TH2F *hRH_iphiVsieta_fullrange_;
  TH1F *h_bxNumber_;
  TH1F *h_pu_;
  // clusters
  TH1F *hCluTPperEvt_;
  TH1F *hCluTPoverRechit_;
  TH2F *hCluTPvsRechit_;
  TH1F *h_nClu_[5];  
  TH1F *h_nXtals_;
  TH1F *h_etCluTP_;
  // matching cluTP with cluRH
  TH1F* h_deltaR_cluTPcluRH_;
  TH1F* h_dEta_cluTP_cluRH_;
  TH1F* h_dPhi_cluTP_cluRH_;
  TH1F* h_fBrem_truth_;

  // matching cluTP with gen electrons
  TH1F* h_deltaR_cluTPGen_;
  TH1F* h_cluTPEtoverGenEt_;
  TH1F* h_dEta_cluTP_gen_;
  TH1F* h_dPhi_cluTP_gen_;

  // matching cluRH with gen ele
  TH1F* h_deltaR_cluRHGen_;
  TH1F* h_dEta_cluRH_gen_;
  TH1F* h_dPhi_cluRH_gen_;
  TH1F* h_cluRHEt_;
  TH1F* h_cluRHEtoverGenEt_;
  // matching gsfEle with gen ele
  TH1F* h_elePtRecoOverPtTrue_;
  TH1F* h_corrEleEtRecoOverPtTrue_;
  TH1F* h_uncorrEleEtRecoOverPtTrue_;
  TH1F* h_dPhi_5x5SC_gen_;
  TH1F* h_5x5SCOverPtTrue_;
  TH1F* h_3x3SCOverPtTrue_;
  TH1F* h_dEta_gsfEle_gen_;
  TH1F* h_dPhi_gsfEle_gen_;
  TH1F* h_deltaR_recoEleGen_;
  // 
  TH2F* h2_recEle_vs_Gen_size_;
  TH2F* h2_cluTP_vs_Gen_size_;
  TH2F* h2_cluRH_vs_Gen_size_;




  TTree *tree_ ;
  TTree *treeCl_;  

  int tpIphi_, tpIeta_ , tpgADC_, ttf_, fg_ ;
  int rhIeta_, rhIphi_;
  float eRec_, tpgGeV_ ;

  int nCl_, etClInADC_;
  float eCl_, etClInGeV_,etaCl_, phiCl_, s_eCl_, s_etCl_;
  float s_etaCl_, s_phiCl_;
  int nXtals_;
  float etCluFromRH_;

  //edm::InputTag label_;
  edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> primToken_;
  edm::EDGetTokenT<EcalEBClusterTrigPrimDigiCollection> primCluToken_;
  edm::EDGetTokenT<EcalRecHitCollection> rechits_labelEB_;
  edm::EDGetTokenT<EBDigiCollection> tokenEBdigi_;
  edm::EDGetTokenT< std::vector<PileupSummaryInfo> > pileupSummaryToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genPartToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectrons_;
  edm::EDGetTokenT<reco::PhotonCollection> gedPhotons_;
  edm::EDGetTokenT<edm::SimTrackContainer>  g4_simTk_Token_;
  edm::EDGetTokenT<edm::SimVertexContainer> g4_simVtx_Token_;

  bool recoContent_;
  bool recHits_;
  bool analyzeElectrons_;
  bool isGenParticleValid_;
  bool debug_;
  double etCluTPThreshold_;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;


  int getIndex(const  EBDigiCollection *, EcalTrigTowerDetId& id) {return id.hashedIndex();}  

};

class SimpleCluster   {

 public:
  SimpleCluster (float et) {  et_=et;}
  void setEta(float eta) {eta_=eta;}
  void setPhi(float phi) {phi_=phi;}
  float eta()  const {return eta_;}
  float phi()  const {return phi_;}
  float et() const {return et_;}

 private:
  float et_;
  float eta_;
  float phi_;

};


class SimpleCaloHit2   {

 public:

  

  SimpleCaloHit2 (float et) {  et_=et;}
  bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
  EBDetId id() const {return id_;}
  void setId(const EBDetId id)  { id_=id;}
  GlobalVector position() {return position_;}
  void setPosition( GlobalVector pos ) {position_=pos;}

  float et() const {return et_;} 
  float energy() {return et()/sin(position().theta());}
  


  int dieta(SimpleCaloHit2& other) const
  {
    
    if (id().ieta() * other.id().ieta() > 0)
      return id().ieta()-other.id().ieta();
    return  id().ieta()-other.id().ieta()-1;
  };
  inline float dphi(SimpleCaloHit2& other) {return reco::deltaPhi(static_cast<float>(position().phi()), static_cast<float>(other.position().phi()));};

  int diphi(SimpleCaloHit2& other) const
  {
    // Logic from EBDetId::distancePhi() without the abs()
    int PI = 180;
    int result = id().iphi() - other.id().iphi();
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
  };
  
  bool operator==(SimpleCaloHit2& other) 
  {
    if ( id() == other.id() &&
	 position() == other.position() &&
	 energy() == other.energy()
	 ) return true;
    
    return false;
  };
  



 private:
  EBDetId id_;
  GlobalVector position_; // As opposed to GlobalPoint, so we can add them (for weighted average)
  float et_;
  
  
};

