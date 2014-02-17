// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzer
//
/**\class EcalTrigPrimAnalyzer

 Description: test of the output of EcalTrigPrimProducer

*/
//
//
// Original Author:  Ursula Berthon, Stephanie Baffioni, Pascal Paganini
//         Created:  Thu Jul 4 11:38:38 CEST 2005
// $Id: EcalTrigPrimAnalyzer.cc,v 1.18 2008/06/24 13:03:14 uberthon Exp $
//
//


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
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "EcalTrigPrimAnalyzer.h"

#include <TMath.h>

using namespace edm;
class CaloSubdetectorGeometry;

EcalTrigPrimAnalyzer::EcalTrigPrimAnalyzer(const edm::ParameterSet&  iConfig)

{
  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  histfile_=new TFile("histos.root","RECREATE");
  tree_ = new TTree("TPGtree","TPGtree");
  tree_->Branch("iphi",&iphi_,"iphi/I");
  tree_->Branch("ieta",&ieta_,"ieta/I");
  tree_->Branch("eRec",&eRec_,"eRec/F");
  tree_->Branch("tpgADC",&tpgADC_,"tpgADC/I");
  tree_->Branch("tpgGeV",&tpgGeV_,"tpgGeV/F");
  tree_->Branch("ttf",&ttf_,"ttf/I");
  tree_->Branch("fg",&fg_,"fg/I");
  for (unsigned int i=0;i<2;++i) {
    ecal_et_[i]=new TH1I(ecal_parts_[i].c_str(),"Et",255,0,255);
    char title[30];
    sprintf(title,"%s_ttf",ecal_parts_[i].c_str());
    ecal_tt_[i]=new TH1I(title,"TTF",10,0,10);
    sprintf(title,"%s_fgvb",ecal_parts_[i].c_str());
    ecal_fgvb_[i]=new TH1I(title,"FGVB",10,0,10);
  }

  recHits_= iConfig.getParameter<bool>("AnalyzeRecHits");
  label_=iConfig.getParameter<edm::InputTag>("inputTP");
  if (recHits_) {
    hTPvsRechit_= new TH2F("TP_vs_RecHit","TP vs  rechit",256,-1,255,255,0,255);
    hTPoverRechit_= new TH1F("TP_over_RecHit","TP over rechit",500,0,4);
    rechits_labelEB_=iConfig.getParameter<edm::InputTag>("inputRecHitsEB");
    rechits_labelEE_=iConfig.getParameter<edm::InputTag>("inputRecHitsEE");
  }
}


EcalTrigPrimAnalyzer::~EcalTrigPrimAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  histfile_->Write();
  histfile_->Close();

}


//
// member functions
//

// ------------ method called to analyze the data  ------------
void
EcalTrigPrimAnalyzer::analyze(const edm::Event& iEvent, const  edm::EventSetup & iSetup)
{
  using namespace edm;
  using namespace std;

  // Get input
  edm::Handle<EcalTrigPrimDigiCollection> tp;
  iEvent.getByLabel(label_,tp);
  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
    int subdet=d.id().subDet()-1;
    if (subdet==0) {
      ecal_et_[subdet]->Fill(d.compressedEt());
    }
    else {
      if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28) {
	if (i%2) ecal_et_[subdet]->Fill(d.compressedEt()*2.);
      }
      else ecal_et_[subdet]->Fill(d.compressedEt());
    }
    ecal_tt_[subdet]->Fill(d.ttFlag());
    ecal_fgvb_[subdet]->Fill(d.fineGrain());

  }
  if (!recHits_) return;

  // comparison with RecHits
  edm::Handle<EcalRecHitCollection> rechit_EB_col;
  iEvent.getByLabel(rechits_labelEB_,rechit_EB_col);

  edm::Handle<EcalRecHitCollection> rechit_EE_col;
  iEvent.getByLabel(rechits_labelEE_,rechit_EE_col);
  

  edm::ESHandle<CaloGeometry> theGeometry;
  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
  iSetup.get<CaloGeometryRecord>().get( theGeometry );
  iSetup.get<EcalEndcapGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  iSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);

  const CaloSubdetectorGeometry *theEndcapGeometry,*theBarrelGeometry;
  theEndcapGeometry = &(*theEndcapGeometry_handle);
  theBarrelGeometry = &(*theBarrelGeometry_handle);
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  iSetup.get<IdealGeometryRecord>().get(eTTmap_);

  map<EcalTrigTowerDetId, float> mapTow_Et;


  for (unsigned int i=0;i<rechit_EB_col.product()->size();i++) {
    const EBDetId & myid1=(*rechit_EB_col.product())[i].id();
    EcalTrigTowerDetId towid1= myid1.tower();
    float theta =  theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
    float Etsum=((*rechit_EB_col.product())[i].energy())*sin(theta);
    bool test_alreadyin= false;
    map<EcalTrigTowerDetId, float>::iterator ittest=  mapTow_Et.find(towid1);
    if (ittest!= mapTow_Et.end()) test_alreadyin=true;
    if (test_alreadyin) continue;
    unsigned int j=i+1;
    bool loopend=false;
    unsigned int count=0;
    while( j<rechit_EB_col.product()->size() && !loopend){
      count++;
      const EBDetId & myid2=(*rechit_EB_col.product())[j].id();
      EcalTrigTowerDetId towid2= myid2.tower();
      if( towid1==towid2 ) {
	float  theta=theBarrelGeometry->getGeometry(myid2)->getPosition().theta();
	Etsum += (*rechit_EB_col.product())[j].energy()*sin(theta);
      }
      j++;
      if (count>1800) loopend=true;
    }
    mapTow_Et.insert(pair<EcalTrigTowerDetId,float>(towid1, Etsum));
  }


  for (unsigned int i=0;i<rechit_EE_col.product()->size();i++) {
    const EEDetId & myid1=(*rechit_EE_col.product())[i].id();
    EcalTrigTowerDetId towid1= (*eTTmap_).towerOf(myid1);
    float  theta=theEndcapGeometry->getGeometry(myid1)->getPosition().theta();
    float Etsum=(*rechit_EE_col.product())[i].energy()*sin(theta);
    bool test_alreadyin= false;
    map<EcalTrigTowerDetId, float>::iterator ittest=  mapTow_Et.find(towid1);
    if (ittest!= mapTow_Et.end()) test_alreadyin=true;
    if (test_alreadyin) continue;
    unsigned int j=i+1;
    bool loopend=false;
    unsigned int count=0;
    while( j<rechit_EE_col.product()->size() && !loopend){
      const EEDetId & myid2=(*rechit_EE_col.product())[j].id();
      EcalTrigTowerDetId towid2= (*eTTmap_).towerOf(myid2);
      if( towid1==towid2 ) {
	float  theta=theEndcapGeometry->getGeometry(myid2)->getPosition().theta();
	Etsum += (*rechit_EE_col.product())[j].energy()*sin(theta);
      }
      //  else loopend=true;
      j++;
      if (count>500) loopend=true;
    }
    //    alreadyin_EE.push_back(towid1);
    mapTow_Et.insert(pair<EcalTrigTowerDetId,float>(towid1, Etsum));
  }


  EcalTPGScale ecalScale ;
  ecalScale.setEventSetup(iSetup) ;
  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
    const EcalTrigTowerDetId TPtowid= d.id();
    map<EcalTrigTowerDetId, float>::iterator it=  mapTow_Et.find(TPtowid);
    float Et = ecalScale.getTPGInGeV(d.compressedEt(), TPtowid) ; 
    if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28)    Et*=2;
    iphi_ = TPtowid.iphi() ;
    ieta_ = TPtowid.ieta() ;
    tpgADC_ = d.compressedEt() ;
    tpgGeV_ = Et ;
    ttf_ = d.ttFlag() ;
    fg_ = d.fineGrain() ;
    if (it!= mapTow_Et.end()) {
      hTPvsRechit_->Fill(it->second,Et);
      hTPoverRechit_->Fill(Et/it->second);
      eRec_ = it->second ;
    }
    tree_->Fill() ;
  }

}

void
EcalTrigPrimAnalyzer::endJob(){
  for (unsigned int i=0;i<2;++i) {
    ecal_et_[i]->Write();
    ecal_tt_[i]->Write();
    ecal_fgvb_[i]->Write();
  }
  if (recHits_) {
    hTPvsRechit_->Write();
    hTPoverRechit_->Write();
  }
}
  
