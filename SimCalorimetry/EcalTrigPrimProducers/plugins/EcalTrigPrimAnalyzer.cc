// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzer
// 
/**\class EcalTrigPrimAnalyzer

 Description: test of the output of EcalTrigPrimProducer

*/
//
//
// Original Author:  Ursula Berthon
//         Created:  Thu Jul 4 11:38:38 CEST 2005
// $Id: EcalTrigPrimAnalyzer.cc,v 1.4 2007/02/15 12:59:24 uberthon Exp $
//
//


// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "EcalTrigPrimAnalyzer.h"

#include <TMath.h>

using namespace edm;
class CaloSubdetectorGeometry;

EcalTrigPrimAnalyzer::EcalTrigPrimAnalyzer(const edm::ParameterSet& iConfig)

{
  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  histfile_=new TFile("histos.root","RECREATE");
  for (unsigned int i=0;i<2;++i) {
    ecal_et_[i]=new TH1I(ecal_parts_[i].c_str(),"Et",255,0,255);
    char title[30];
    sprintf(title,"%s_ttf",ecal_parts_[i].c_str());
    ecal_tt_[i]=new TH1I(title,"TTF",10,0,10);
    sprintf(title,"%s_fgvb",ecal_parts_[i].c_str());
    ecal_fgvb_[i]=new TH1I(title,"FGVB",10,0,10);
  }
  hTPvsRechit_= new TH2F("TP_vs_RecHit","TP vs rechit",256,-1,255,255,0,255);
  hTPoverRechit_= new TH1F("TP_over_RecHit","TP over rechit",500,0,4);
  label_= iConfig.getParameter<std::string>("Label");
  producer_= iConfig.getParameter<std::string>("Producer");
  rechits_labelEB_= iConfig.getParameter<std::string>("RecHitsLabelEB");
  rechits_labelEE_= iConfig.getParameter<std::string>("RecHitsLabelEE");
  rechits_producer_= iConfig.getParameter<std::string>("RecHitsProducer");}


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
EcalTrigPrimAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup & iSetup)
{
  using namespace edm;
  using namespace std;

  // Get input
  edm::Handle<EcalTrigPrimDigiCollection> tp;
  iEvent.getByLabel(label_,producer_,tp);
  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
    int subdet=d.id().subDet()-1;
    if (subdet==0) {
      ecal_et_[subdet]->Fill(d.compressedEt());
    }
    else {  // here we recombine the 2 virtual towers
      if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28) {
	if (i%2) ecal_et_[subdet]->Fill(d.compressedEt()*2.);
      }
      else ecal_et_[subdet]->Fill(d.compressedEt());
    }
    ecal_tt_[subdet]->Fill(d.ttFlag());
    ecal_fgvb_[subdet]->Fill(d.fineGrain());

  }

  edm::Handle<EcalRecHitCollection> rechit_EB_col;
  iEvent.getByLabel(rechits_producer_,rechits_labelEB_,rechit_EB_col);

  edm::Handle<EcalRecHitCollection> rechit_EE_col;
  iEvent.getByLabel(rechits_producer_,rechits_labelEE_, rechit_EE_col);

  edm::ESHandle<CaloGeometry> theGeometry;
  edm::ESHandle<CaloSubdetectorGeometry> theBarrelGeometry_handle;
  edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle;
  iSetup.get<IdealGeometryRecord>().get( theGeometry );
  iSetup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  iSetup.get<IdealGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
    
  const CaloSubdetectorGeometry *theEndcapGeometry,*theBarrelGeometry;
  theEndcapGeometry = &(*theEndcapGeometry_handle);
  theBarrelGeometry = &(*theBarrelGeometry_handle);
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  iSetup.get<IdealGeometryRecord>().get(eTTmap_);
  
  map<EcalTrigTowerDetId, float> mapTow_Et;
  
  vector<EcalTrigTowerDetId> alreadyin_EB;
  for (unsigned int i=0;i<rechit_EB_col.product()->size();i++) {
    const EBDetId & myid1=(*rechit_EB_col.product())[i].id();

    EcalTrigTowerDetId towid1= myid1.tower();

    float theta = theBarrelGeometry->getGeometry(myid1)->getPosition().theta();
    float Etsum=((*rechit_EB_col.product())[i].energy())*sin(theta);

    bool test_alreadyin= false; 
    unsigned int size=alreadyin_EB.size();
    if (size!=0 && towid1 == alreadyin_EB[(size-1)]) test_alreadyin=true;

    if (test_alreadyin) continue;
    unsigned int j=i+1;
    bool loopend=false;
    unsigned int count=0;
    while( j<rechit_EB_col.product()->size() && !loopend){
      count++;
      const EBDetId & myid2=(*rechit_EB_col.product())[j].id();
      EcalTrigTowerDetId towid2= myid2.tower();
      if( towid1==towid2 ) {
	float theta=theBarrelGeometry->getGeometry(myid2)->getPosition().theta();
	Etsum += (*rechit_EB_col.product())[j].energy()*sin(theta);
      }

      j++;
      if (count>360) loopend=true;
      
    }
    alreadyin_EB.push_back(towid1);
    mapTow_Et.insert(pair<EcalTrigTowerDetId,float>(towid1, Etsum));
  }
  

  
  vector<EcalTrigTowerDetId> alreadyin_EE;
  for (unsigned int i=0;i<rechit_EE_col.product()->size();i++) {
    const EEDetId & myid1=(*rechit_EE_col.product())[i].id();
    EcalTrigTowerDetId towid1= (*eTTmap_).towerOf(myid1);
    float theta=theEndcapGeometry->getGeometry(myid1)->getPosition().theta();
    float Etsum=(*rechit_EE_col.product())[i].energy()*sin(theta);
    bool test_alreadyin= false; 
    unsigned int size=alreadyin_EE.size();
    if (size!=0 && towid1 == alreadyin_EE[(size-1)]) test_alreadyin=true;
    if (test_alreadyin) continue;
    unsigned int j=i+1;
    bool loopend=false;
    unsigned int count=0;
    while( j<rechit_EE_col.product()->size() && !loopend){
      const EEDetId & myid2=(*rechit_EE_col.product())[j].id();
      EcalTrigTowerDetId towid2= (*eTTmap_).towerOf(myid2);
      if( towid1==towid2 ) {
	float theta=theEndcapGeometry->getGeometry(myid2)->getPosition().theta();
	Etsum += (*rechit_EE_col.product())[j].energy()*sin(theta);
      }
      j++;
      if (count>360) loopend=true;
    }
    mapTow_Et.insert(pair<EcalTrigTowerDetId,float>(towid1, Etsum));
  }

  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
    const EcalTrigTowerDetId TPtowid= d.id();
    map<EcalTrigTowerDetId, float>::iterator it= mapTow_Et.find(TPtowid);

    if (it!= mapTow_Et.end()) {
      int subdet=d.id().subDet()-1;
      float Et=d.compressedEt();
      if (subdet==0) Et*=0.469;
      if (subdet==1) Et*=0.56;
      if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28) { // here we recombine the 2 virtual towers
	hTPvsRechit_->Fill(it->second,Et*2);
	hTPoverRechit_->Fill(Et*2/it->second);
      }
      else { 
	hTPvsRechit_->Fill(it->second,Et);
	hTPoverRechit_->Fill(Et/it->second);
      }
      if( (Et< (0.9* it->second) || Et> (1.1* it->second))  && it->second >0.4){
      }
    }
  }
}
    


