// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#include "CalibCalorimetry/EcalTPGTools/interface/EcalEBTPGScale.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "EcalEBTrigPrimAnalyzer.h"

#include <TMath.h>

using namespace edm;
class CaloSubdetectorGeometry;

EcalEBTrigPrimAnalyzer::EcalEBTrigPrimAnalyzer(const edm::ParameterSet& iConfig)

{
  ecal_parts_.push_back("Barrel");

  histfile_ = new TFile("histos.root", "RECREATE");
  tree_ = new TTree("TPGtree", "TPGtree");
  tree_->Branch("tpIphi", &tpIphi_, "tpIphi/I");
  tree_->Branch("tpIeta", &tpIeta_, "tpIeta/I");
  tree_->Branch("rhIphi", &rhIphi_, "rhIphi/I");
  tree_->Branch("rhIeta", &rhIeta_, "rhIeta/I");
  tree_->Branch("eRec", &eRec_, "eRec/F");
  tree_->Branch("tpgADC", &tpgADC_, "tpgADC/I");
  tree_->Branch("tpgGeV", &tpgGeV_, "tpgGeV/F");
  tree_->Branch("ttf", &ttf_, "ttf/I");
  tree_->Branch("fg", &fg_, "fg/I");
  for (unsigned int i = 0; i < ecal_parts_.size(); ++i) {
    char title[30];
    sprintf(title, "%s_Et", ecal_parts_[i].c_str());
    ecal_et_[i] = new TH1I(title, "Et", 255, 0, 255);
    sprintf(title, "%s_ttf", ecal_parts_[i].c_str());
    ecal_tt_[i] = new TH1I(title, "TTF", 10, 0, 10);
    sprintf(title, "%s_fgvb", ecal_parts_[i].c_str());
    ecal_fgvb_[i] = new TH1I(title, "FGVB", 10, 0, 10);
  }

  recHits_ = iConfig.getParameter<bool>("AnalyzeRecHits");
  debug_ = iConfig.getParameter<bool>("Debug");
  rechits_labelEB_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("inputRecHitsEB"));
  primToken_ = consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("inputTP"));
  tokenEBdigi_ = consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("barrelEcalDigis"));
  nEvents_ = 0;

  hTPvsTow_eta_ = new TH2F("TP_vs_Tow_eta", "TP vs Tow eta ; #eta(tow); #eta(tp)", 50, -2.5, 2.5, 50, -2.5, 2.5);
  hAllTPperEvt_ = new TH1F("AllTPperEvt", "TP per Event; N_{TP};  ", 100, 0., 20000.);
  hTPperEvt_ = new TH1F("TPperEvt", "N_{TP} per Event; N_{TP};  ", 100, 0., 500.);
  hTP_iphiVsieta_ = new TH2F("TP_iphiVsieta", "TP i#phi vs i#eta ; i#eta(tp); i#phi(tp)", 10, 70, 80, 10, 340, 350);
  hTP_iphiVsieta_fullrange_ =
      new TH2F("TP_iphiVsieta_fullrange", "TP i#phi vs i#eta ; i#eta(tp); i#phi(tp)", 200, -100, 100, 350, 0, 350);

  if (recHits_) {
    hTPvsTow_ieta_ =
        new TH2F("TP_vs_Tow_ieta", "TP vs Tow ieta ; i#eta(tow); i#eta(tp)", 200, -100, 100, 200, -100, 100);

    hTPvsRechit_ = new TH2F("TP_vs_RecHit", "TP vs rechit Et;E_{T}(rh) (GeV);E_{T}(tp) (GeV)", 100, 0, 50, 100, 0, 50);
    hDeltaEt_ = new TH1F("DeltaEt", "[Et(rh)-Et(TP)]/Et(rh); [E_{T}(rh)-E_{T}(tp)]/E_{T}(rh); Counts", 200, -1, 1);
    hTPoverRechit_ = new TH1F("TP_over_RecHit", "Et(TP/rechit); E_{T}(tp)/E_{T}(rh); Counts", 200, 0, 2);
    hRechitEt_ = new TH1F("RecHitEt", "E_{T};E_{T}(rh) (GeV);Counts", 100, 0, 50);
    hTPEt_ = new TH1F("TPEt", "E_{T}{tp);E_{T}(rh) (GeV);Count", 100, 0, 50);
    hRatioEt_ = new TH1F("RatioTPoverRH", "Et", 100, 0, 50);
    hAllRechitEt_ = new TH1F("AllRecHit", "Et", 100, 0, 50);

    hRH_iphiVsieta_ = new TH2F("RH_iphiVsieta", "RH i#phi vs i#eta ; i#eta(rh); i#phi(rh)", 10, 70, 80, 10, 340, 350);
    hRH_iphiVsieta_fullrange_ =
        new TH2F("RH_iphiVsieta_fullrange", "RH i#phi vs i#eta ; i#eta(rh); i#phi(rh)", 200, -100, 100, 350, 0, 350);
  }
}

EcalEBTrigPrimAnalyzer::~EcalEBTrigPrimAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  histfile_->Write();
  histfile_->Close();
}

void EcalEBTrigPrimAnalyzer::init(const edm::EventSetup& iSetup) { iSetup.get<IdealGeometryRecord>().get(eTTmap_); }

//
// member functions
//

// ------------ method called to analyze the data  ------------
void EcalEBTrigPrimAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  nEvents_++;

  if (nEvents_ == 1)
    this->init(iSetup);

  // Get input
  edm::Handle<EcalEBTrigPrimDigiCollection> tp;
  iEvent.getByToken(primToken_, tp);
  //
  /*
  edm::Handle<EBDigiCollection> barrelDigiHandle;
  const EBDigiCollection *ebdigi=NULL;
  iEvent.getByToken(tokenEBdigi_,barrelDigiHandle);
  ebdigi=barrelDigiHandle.product();
  */

  for (unsigned int i = 0; i < tp.product()->size(); i++) {
    EcalEBTriggerPrimitiveDigi d = (*(tp.product()))[i];
    int subdet = 0;
    if (subdet == 0) {
      ecal_et_[subdet]->Fill(d.encodedEt());
    }
  }

  //  if (!recHits_) return;

  edm::Handle<EcalRecHitCollection> rechit_EB_col;
  if (recHits_) {
    // get the  RecHits
    iEvent.getByToken(rechits_labelEB_, rechit_EB_col);
  }

  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  iSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel", theBarrelGeometry_handle);
  const CaloSubdetectorGeometry* theBarrelGeometry = theBarrelGeometry_handle.product();

  if (debug_) {
    std::cout << " TP analyzer  =================> Treating event  " << iEvent.id() << " Number of TPs "
              << tp.product()->size() << std::endl;
    if (recHits_)
      std::cout << " Number of EB rechits " << rechit_EB_col.product()->size() << std::endl;
  }
  hAllTPperEvt_->Fill(float(tp.product()->size()));

  //if ( iEvent.id().event() != 648) return;

  //EcalEBTPGScale ecalScale ;
  EcalTPGScale ecalScale;
  ecalScale.setEventSetup(iSetup);

  //  for(unsigned int iDigi = 0; iDigi < ebdigi->size() ; ++iDigi) {
  // EBDataFrame myFrame((*ebdigi)[iDigi]);
  // const EBDetId & myId = myFrame.id();

  int nTP = 0;
  for (unsigned int i = 0; i < tp.product()->size(); i++) {
    EcalEBTriggerPrimitiveDigi d = (*(tp.product()))[i];
    const EBDetId TPid = d.id();
    // if ( myId != TPid ) continue;

    /*    
	  int index=getIndex(ebdigi,coarser);
	  std::cout << " Same xTal " << myId << " " << TPid << " coarser " << coarser << " index " << index << std::endl;
	  double Et = ecalScale.getTPGInGeV(d.encodedEt(), coarser) ; 
    */
    //this works if the energy is compressed into 8 bits float Et=d.compressedEt()/2.; // 2ADC counts/GeV
    float Et = d.encodedEt() / 8.;  // 8 ADCcounts/GeV
    if (Et <= 5)
      continue;
    //if ( Et<= 0 ) continue;
    nTP++;

    std::cout << " TP digi size " << d.size() << std::endl;
    for (int iBx = 0; iBx < d.size(); iBx++) {
      std::cout << " TP samples " << d.sample(iBx) << std::endl;
    }

    //      EcalTrigTowerDetId coarser=(*eTTmap_).towerOf(myId);
    // does not work float etaTow =  theBarrelGeometry->getGeometry(coarser)->getPosition().theta();
    // float etaTP =  theBarrelGeometry->getGeometry(TPid)->getPosition().eta();
    // does not work hTPvsTow_eta_->Fill ( etaTow,  etaTP );
    //      hTPvsTow_ieta_->Fill ( coarser.ieta(),  TPid.ieta() );

    tpIphi_ = TPid.iphi();
    tpIeta_ = TPid.ieta();
    tpgADC_ = d.encodedEt();
    tpgGeV_ = Et;

    hTP_iphiVsieta_->Fill(TPid.ieta(), TPid.iphi(), Et);
    hTP_iphiVsieta_fullrange_->Fill(TPid.ieta(), TPid.iphi(), Et);

    if (recHits_) {
      for (unsigned int j = 0; j < rechit_EB_col.product()->size(); j++) {
        const EBDetId& myid1 = (*rechit_EB_col.product())[j].id();
        float theta = theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
        float rhEt = ((*rechit_EB_col.product())[j].energy()) * sin(theta);
        if (myid1 == TPid) {
          if (debug_)
            std::cout << " Analyzer same cristal " << myid1 << " " << TPid << std::endl;
          //	  if ( rhEt < 1.5 && Et > 10 )  {
          // std::cout << " TP analyzer  =================> Treating event  "<<iEvent.id()<< ", Number of EB rechits "<<  rechit_EB_col.product()->size() <<  " Number of TPs " <<  tp.product()->size() <<  std::endl;
          //std::cout << " TP compressed et " << d.encodedEt()  << " Et in GeV  " <<  Et << " RH Et " << rhEt << " Et/rhEt " << Et/rhEt << std::endl;
          //}

          //std::cout << " TP out " <<  d << std::endl;

          //	  for (int isam=0;isam< d.size();++isam) {
          // std::cout << " d[isam].raw() "  <<  d[isam].raw() << std::endl;
          //}

          rhIphi_ = myid1.iphi();
          rhIeta_ = myid1.ieta();
          hRH_iphiVsieta_->Fill(myid1.ieta(), myid1.iphi(), rhEt);
          hRH_iphiVsieta_fullrange_->Fill(myid1.ieta(), myid1.iphi(), rhEt);

          hTPvsRechit_->Fill(rhEt, Et);
          hTPoverRechit_->Fill(Et / rhEt);
          hDeltaEt_->Fill((rhEt - Et) / rhEt);
          if (debug_)
            std::cout << " TP compressed et " << d.encodedEt() << " Et in GeV  " << Et << " RH Et " << rhEt
                      << " Et/rhEt " << Et / rhEt << std::endl;
          hRechitEt_->Fill(rhEt);
          hTPEt_->Fill(Et);
          if (rhEt < 1000000)
            eRec_ = rhEt;
          tree_->Fill();
        }

      }  // end loop of recHits
    }    // if recHits

  }  // end loop over TP collection

  //  } // end loop over digi collection

  hTPperEvt_->Fill(float(nTP));

  if (recHits_) {
    hRatioEt_->Divide(hTPEt_, hRechitEt_);
    for (unsigned int j = 0; j < rechit_EB_col.product()->size(); j++) {
      const EBDetId& myid1 = (*rechit_EB_col.product())[j].id();
      float theta = theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
      float rhEt = ((*rechit_EB_col.product())[j].energy()) * sin(theta);
      if (rhEt > 0)
        hAllRechitEt_->Fill(rhEt);
    }
  }
}

void EcalEBTrigPrimAnalyzer::endJob() {
  for (unsigned int i = 0; i < ecal_parts_.size(); ++i) {
    ecal_et_[i]->Write();
    ecal_tt_[i]->Write();
    ecal_fgvb_[i]->Write();
  }

  hAllTPperEvt_->Write();
  hTPperEvt_->Write();

  if (recHits_) {
    hTPvsTow_ieta_->Write();
    hTPvsTow_eta_->Write();
    hTPvsRechit_->Write();
    hTPoverRechit_->Write();
    hAllRechitEt_->Write();
    hRechitEt_->Write();
    hDeltaEt_->Write();
    hTPEt_->Write();
    hRatioEt_->Write();
    hTP_iphiVsieta_->Write();
    hRH_iphiVsieta_->Write();
    hTP_iphiVsieta_fullrange_->Write();
    hRH_iphiVsieta_fullrange_->Write();
  }
}
