#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TH1F.h>

//
// class declaration
//

class EcalEBTrigPrimAnalyzer : public edm::one::EDAnalyzer<> {
   public:
  explicit EcalEBTrigPrimAnalyzer(const edm::ParameterSet&);
  ~EcalEBTrigPrimAnalyzer();


  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();
   private:

  int nEvents_;
  void init(const edm::EventSetup&);

  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I * ecal_et_[2];
  TH1I * ecal_tt_[2];
  TH1I * ecal_fgvb_[2];
  TH1I *histEndc,*histBar;
  TFile *histfile_;
  TH2F *hTPvsRechit_;
  TH1F *hTPoverRechit_;
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

  TTree *tree_ ;

  int tpIphi_, tpIeta_ , tpgADC_, ttf_, fg_ ;
  int rhIeta_, rhIphi_;
  float eRec_, tpgGeV_ ;

  //edm::InputTag label_;
  edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> primToken_;
  edm::EDGetTokenT<EcalRecHitCollection> rechits_labelEB_;
  edm::EDGetTokenT<EBDigiCollection> tokenEBdigi_;
  bool recHits_;
  bool debug_;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;

  int getIndex(const  EBDigiCollection *, EcalTrigTowerDetId& id) {return id.hashedIndex();}  

};
