// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzerMIPs
//
//
// Original Author:  Pascal Paganini
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
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "EcalTrigPrimAnalyzerMIPs.h"

#include <TMath.h>

using namespace edm;
class CaloSubdetectorGeometry;

EcalTrigPrimAnalyzerMIPs::EcalTrigPrimAnalyzerMIPs(const edm::ParameterSet&  iConfig)
  : nevt_(0)
{
  label_= iConfig.getParameter<std::string>("Label");
  producer_= iConfig.getParameter<std::string>("Producer");
  digi_label_= iConfig.getParameter<std::string>("DigiLabel");
  digi_producer_=  iConfig.getParameter<std::string>("DigiProducer");
  emul_label_= iConfig.getParameter<std::string>("EmulLabel");
  emul_producer_=  iConfig.getParameter<std::string>("EmulProducer");


  histfile_ = new TFile("histos.root","RECREATE");

  // general tree
  tree_ = new TTree("TPGtree","TPGtree");
  tree_->Branch("iphi",&iphi_,"iphi/I");
  tree_->Branch("ieta",&ieta_,"ieta/I");
  tree_->Branch("eRec",&eRec_,"eRec/F");
  tree_->Branch("mean",&mean_,"mean/F");
  tree_->Branch("data0",&data0_,"data0/F");
  tree_->Branch("data1",&data1_,"data1/F");
  tree_->Branch("data2",&data2_,"data2/F");
  tree_->Branch("data3",&data3_,"data3/F");
  tree_->Branch("data4",&data4_,"data4/F");
  tree_->Branch("data5",&data5_,"data5/F");
  tree_->Branch("data6",&data6_,"data6/F");
  tree_->Branch("data7",&data7_,"data7/F");
  tree_->Branch("data8",&data8_,"data8/F");
  tree_->Branch("data9",&data9_,"data9/F");
  tree_->Branch("tpgADC",&tpgADC_,"tpgADC/I");
  tree_->Branch("tpgGeV",&tpgGeV_,"tpgGeV/F");
  tree_->Branch("tpgEmul0",&tpgEmul0_,"tpgEmul0/I");
  tree_->Branch("tpgEmul1",&tpgEmul1_,"tpgEmul1/I");
  tree_->Branch("tpgEmul2",&tpgEmul2_,"tpgEmul2/I");
  tree_->Branch("tpgEmul3",&tpgEmul3_,"tpgEmul3/I");
  tree_->Branch("tpgEmul4",&tpgEmul4_,"tpgEmul4/I");
  tree_->Branch("ttf",&ttf_,"ttf/I");
  tree_->Branch("fg",&fg_,"fg/I");
  tree_->Branch("nevt",&nevt_,"nevt/I");
  tree_->Branch("nXtal",&nXtal_,"nXtal/I");
  tree_->Branch("sample",&sample_,"sample/F");

  // tree to analyze missing FEDs 
  fedtree_ = new TTree("fedtree","fedtree");
  fedtree_->Branch("fedId",&fedId_,"fedId/I");
  fedtree_->Branch("fedSize",&fedSize_,"fedSize/I");

  // tree for TOP-Bottom coincidence
  treetopbot_ = new TTree("topbottree", "topbottree") ;
  treetopbot_->Branch("nevt",&nevt_,"nevt/I");
  treetopbot_->Branch("iphitop",&iphitop_,"iphitop/I");
  treetopbot_->Branch("ietatop",&ietatop_,"ietatop/I");
  treetopbot_->Branch("Etop",&Etop_,"Etop/F");
  treetopbot_->Branch("Ntop",&Ntop_,"Ntop/I");
  treetopbot_->Branch("iphibot",&iphibot_,"iphibot/I");
  treetopbot_->Branch("ietabot",&ietabot_,"ietabot/I");
  treetopbot_->Branch("Ebot",&Ebot_,"Ebot/F");
  treetopbot_->Branch("Nbot",&Nbot_,"Nbot/I");

}


EcalTrigPrimAnalyzerMIPs::~EcalTrigPrimAnalyzerMIPs()
{
  histfile_->Write();
  histfile_->Close();
}


//
// member functions
//

// ------------ method called to analyze the data  ------------
void EcalTrigPrimAnalyzerMIPs::analyze(const edm::Event& iEvent, const  edm::EventSetup & iSetup)
{
  using namespace edm;
  using namespace std;



  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByType(rawdata);  
  for (int id= 0; id<=FEDNumbering::lastFEDId(); ++id){ 
    if (id < 600 || id > 654) continue;
    const FEDRawData& data = rawdata->FEDData(id);
    fedId_ = id ;
    fedSize_ =  data.size() ;
    fedtree_->Fill() ;
  }


  map<EcalTrigTowerDetId, towerEner> mapTower ;
  map<EcalTrigTowerDetId, towerEner>::iterator itTT ;
  
  // Get digi input
  edm::Handle<EBDigiCollection> digi;
  iEvent.getByLabel(digi_label_, digi_producer_, digi);
  for (unsigned int i=0;i<digi.product()->size();i++) {
    const EBDataFrame & df = (*(digi.product()))[i];
    
    int gain, adc ;
    float E_xtal = 0. ; 
    int theSamp = 0 ;
    float mean = 0., max = -999 ; 
    for (int samp = 0 ; samp<10 ; samp++) {
      adc = df[samp].adc() ;
      if (samp<2) mean += adc ;
      if (adc>max) {
	max = adc ;
	theSamp = samp ;
      }
    }
    mean /= 2 ;
    if (mean>0 && max > mean + 10) {
      gain = df[theSamp].gainId() ;
      adc = df[theSamp].adc() ;
      if (gain == 1) E_xtal = (adc-mean) ;
      if (gain == 2) E_xtal = 2.*(adc-mean) ;
      if (gain == 3) E_xtal = 12.*(adc-mean) ;
      if (gain == 0) E_xtal = 12.*(adc-mean) ;
    }
    const EBDetId & id=df.id();
    const EcalTrigTowerDetId towid= id.tower();
    itTT = mapTower.find(towid) ;
    if (itTT != mapTower.end()) {
      (itTT->second).eRec_ += E_xtal ;
      (itTT->second).sample_ += E_xtal*theSamp ;
      for (int samp = 0 ; samp<10 ; samp++) (itTT->second).data_[samp] += df[samp].adc()-mean ;
      if (E_xtal != 0) {
	(itTT->second).nXtal_ ++ ;
	(itTT->second).mean_ += mean ;
      }
    }
    else {
      towerEner tE ;
      tE.eRec_ = E_xtal ;
      tE.sample_ += E_xtal*theSamp ;
      for (int samp = 0 ; samp<10 ; samp++) tE.data_[samp] = df[samp].adc()-mean ;
      if (E_xtal != 0) {
	tE.nXtal_ ++ ;
	tE.mean_ = mean ;	
      }
      mapTower[towid] = tE ;
    }
  }



  // Get Emulators TP
  edm::Handle<EcalTrigPrimDigiCollection> tpEmul ;
  iEvent.getByLabel(emul_label_, emul_producer_, tpEmul);
  for (unsigned int i=0;i<tpEmul.product()->size();i++) {
    EcalTriggerPrimitiveDigi d = (*(tpEmul.product()))[i];
    const EcalTrigTowerDetId TPtowid= d.id();
    itTT = mapTower.find(TPtowid) ;

    if (itTT != mapTower.end()) {
      (itTT->second).tpgEmul0_ = (d[0].raw() & 0x1ff) ;
      (itTT->second).tpgEmul1_ = (d[1].raw() & 0x1ff) ;
      (itTT->second).tpgEmul2_ = (d[2].raw() & 0x1ff) ;
      (itTT->second).tpgEmul3_ = (d[3].raw() & 0x1ff) ;
      (itTT->second).tpgEmul4_ = (d[4].raw() & 0x1ff) ;
    }
    else {
      towerEner tE ;
      tE.tpgEmul0_ = (d[0].raw() & 0x1ff) ;
      tE.tpgEmul1_ = (d[1].raw() & 0x1ff) ;
      tE.tpgEmul2_ = (d[2].raw() & 0x1ff) ;
      tE.tpgEmul3_ = (d[3].raw() & 0x1ff) ;
      tE.tpgEmul4_ = (d[4].raw() & 0x1ff) ;
      mapTower[TPtowid] = tE ;
    }
  }



  // Get TP data
  edm::Handle<EcalTrigPrimDigiCollection> tp;
  iEvent.getByLabel(label_,producer_,tp);
  EcalTPGScale ecalScale;
  ecalScale.setEventSetup(iSetup) ;
  for (unsigned int i=0;i<tp.product()->size();i++) {
    EcalTriggerPrimitiveDigi d = (*(tp.product()))[i];
    const EcalTrigTowerDetId TPtowid= d.id();
    float Et = ecalScale.getTPGInGeV(d) ; 
    if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28)    Et*=2;

    itTT = mapTower.find(TPtowid) ;
    if (itTT != mapTower.end()) {
      (itTT->second).iphi_ = TPtowid.iphi() ;
      (itTT->second).ieta_ = TPtowid.ieta() ;
      (itTT->second).tpgADC_ = d.compressedEt() ;
      (itTT->second).tpgGeV_ = Et ;
      (itTT->second).ttf_ = d.ttFlag() ;
      (itTT->second).fg_ = d.fineGrain() ;      
    }
    else {
      towerEner tE ;
      tE.iphi_ = TPtowid.iphi() ;
      tE.ieta_ = TPtowid.ieta() ;
      tE.tpgADC_ = d.compressedEt() ;
      tE.tpgGeV_ = Et ;
      tE.ttf_ = d.ttFlag() ;
      tE.fg_ = d.fineGrain() ;    
      mapTower[TPtowid] = tE ;
    }

  }



  // fill tree
  if (mapTower.size()>0) nevt_++ ;
  for (itTT = mapTower.begin() ; itTT != mapTower.end() ; ++itTT ) {
    iphi_ = (itTT->second).iphi_ ;
    ieta_ = (itTT->second).ieta_ ;
    tpgADC_ = (itTT->second).tpgADC_ ;
    tpgGeV_ = (itTT->second).tpgGeV_ ;
    tpgEmul0_ = (itTT->second).tpgEmul0_ ;
    tpgEmul1_ = (itTT->second).tpgEmul1_ ;
    tpgEmul2_ = (itTT->second).tpgEmul2_ ;
    tpgEmul3_ = (itTT->second).tpgEmul3_ ;
    tpgEmul4_ = (itTT->second).tpgEmul4_ ;
    ttf_ = (itTT->second).ttf_ ;
    fg_ = (itTT->second).fg_ ;
    eRec_ = (itTT->second).eRec_ ;
    mean_ = (itTT->second).mean_ ;
    data0_ = (itTT->second).data_[0] ;
    data1_ = (itTT->second).data_[1] ;
    data2_ = (itTT->second).data_[2] ;
    data3_ = (itTT->second).data_[3] ;
    data4_ = (itTT->second).data_[4] ;
    data5_ = (itTT->second).data_[5] ;
    data6_ = (itTT->second).data_[6] ;
    data7_ = (itTT->second).data_[7] ;
    data8_ = (itTT->second).data_[8] ;
    data9_ = (itTT->second).data_[9] ;
    nXtal_ = (itTT->second).nXtal_ ;
    sample_ = 0 ;
    if (eRec_>0) sample_ = (itTT->second).sample_/eRec_ ;
    tree_->Fill() ;

//     int maxtpg = 0 ;
//     if (tpgEmul0_ > tpgEmul1_ && tpgEmul0_ > tpgEmul2_ && tpgEmul0_ > tpgEmul3_ && tpgEmul0_ > tpgEmul4_) maxtpg = tpgEmul0_ ;
//     if (tpgEmul1_ > tpgEmul0_ && tpgEmul1_ > tpgEmul2_ && tpgEmul1_ > tpgEmul3_ && tpgEmul1_ > tpgEmul4_) maxtpg = tpgEmul1_ ;
//     if (tpgEmul2_ > tpgEmul1_ && tpgEmul2_ > tpgEmul0_ && tpgEmul2_ > tpgEmul3_ && tpgEmul2_ > tpgEmul4_) maxtpg = tpgEmul2_ ;
//     if (tpgEmul3_ > tpgEmul1_ && tpgEmul3_ > tpgEmul2_ && tpgEmul3_ > tpgEmul0_ && tpgEmul3_ > tpgEmul4_) maxtpg = tpgEmul3_ ;
//     if (tpgEmul4_ > tpgEmul1_ && tpgEmul4_ > tpgEmul2_ && tpgEmul4_ > tpgEmul3_ && tpgEmul4_ > tpgEmul0_) maxtpg = tpgEmul4_ ;

//     if (maxtpg>=40) {

//       int phiArray[19] = {19, 11, 12, 55, 56, 57, 58, 51, 52, 53, 54, 55, 56, 57, 58, 15, 16, 17, 18} ;
//       int etaArray[19] = {15, 9, 9, 12, 12, 12, 12, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6} ;

//       for (int bad=0 ; bad<19 ; bad++) {
      
// 	if (iphi_==phiArray[bad] && ieta_==etaArray[bad]) {
// 	  std::cout<<"nevt "<<nevt_<<" "<<iphi_<<" "<<ieta_<<std::endl ;
	
// 	  float max = 0. ;
// 	  int xtal_iphi = 0, xtal_ieta = 0, xtal_ic = 0, xtal_sm = 0 ;
// 	  for (unsigned int i=0;i<digi.product()->size();i++) {
// 	    const EBDataFrame & df = (*(digi.product()))[i];
// 	    const EBDetId & id=df.id();
// 	    const EcalTrigTowerDetId towid= id.tower();
// 	    if (towid.iphi()== phiArray[bad] && towid.ieta()== etaArray[bad]) {
	      
// 	      float mean = (df[0].adc()+df[1].adc())/2. ;
// 	      float adc = 0. ;
// 	      for (int s=0 ; s<10 ; s++) if (df[s].adc() > adc) adc = df[s].adc() ;
// 	      adc -= mean ;

// 	      if (adc>max) {
// 		max = adc ;
// 		xtal_iphi = id.iphi() ;
// 		xtal_ieta = id.ieta() ;	    
// 		xtal_ic = id.ic() ;	    
// 		xtal_sm = id.ism() ;	    
// 	      }
	    	      
// 	    }
// 	  }
// 	  std::cout<<xtal_iphi<<" "<<xtal_ieta<<" "<<xtal_ic<<" "<<xtal_sm<<" "<<max<<std::endl ;
// 	}
//       }
//     }

      
  }


// trying to find coincidence :
  float E_max_top = 0.,  E_max_bot = 0.;
  EBDetId idRef_top, idRef_bot ;
  for (unsigned int i=0;i<digi.product()->size();i++) {
    const EBDataFrame & df = (*(digi.product()))[i];
    const EBDetId & id=df.id();
    const EcalTrigTowerDetId towid= id.tower();

    // lets's exclude noisy tower: 
    bool good(true) ;
    if (towid.ieta() == 15 && towid.iphi() == 19) good = false ;
    if (towid.ieta() == 9 && towid.iphi() == 11) good = false ;
    if (towid.ieta() == 9 && towid.iphi() == 12) good = false ;
    if (towid.ieta() == 12 && towid.iphi()>54 && towid.iphi()<59) good = false ;
    if (towid.ieta() == 5 && towid.iphi()>50 && towid.iphi()<55) good = false ;
    if (towid.ieta() == 6 && towid.iphi()>54 && towid.iphi()<59) good = false ;
    if (towid.ieta() == 6 && towid.iphi()>14 && towid.iphi()<19) good = false ;	  

    if (good) {

      // top:
      if (id.ism() >= 4 && id.ism() <= 7) {
	// get the most energetic xtal:
	int adc ;
	float E_xtal = 0. ; 
	float mean = 0.5*(df[0].adc()+df[1].adc()) ; 
	float max = -999 ; 
	for (int samp = 0 ; samp<10 ; samp++) {
	  adc = df[samp].adc() ;
	  if (adc>max) max = adc ;
	}
	if (mean>0 && max > mean + 10) E_xtal = (adc-mean) ;
	if (E_xtal > E_max_top) {
	  E_max_top = E_xtal ;
	  idRef_top = id ;
	}
      }
      
      // bottom:
      if (id.ism() >= 14 && id.ism() <= 16) {
	int adc ;
	float E_xtal = 0. ; 
	float mean = 0.5*(df[0].adc()+df[1].adc()) ; 
	float max = -999 ; 
	for (int samp = 0 ; samp<10 ; samp++) {
	  adc = df[samp].adc() ;
	  if (adc>max) max = adc ;
	}
	if (mean>0 && max > mean + 10) E_xtal = (adc-mean) ;
	if (E_xtal > E_max_bot) {
	  E_max_bot = E_xtal ;
	  idRef_bot = id ;
	}	
      }
      
    }
  }
  if (E_max_top >0 && E_max_bot>0) {
    std::cout<<nevt_<<std::endl ;
    std::cout<<idRef_top.iphi()<<" "<<idRef_top.ieta()<<" "<<idRef_top.ic()<<" "<<idRef_top.ism()<<" "<<E_max_top<<std::endl ;
    std::cout<<idRef_bot.iphi()<<" "<<idRef_bot.ieta()<<" "<<idRef_bot.ic()<<" "<<idRef_bot.ism()<<" "<<E_max_bot<<std::endl ;

    // now lets make a 3x3 window
    int rangePhitop[3] = {idRef_top.iphi()-1, idRef_top.iphi(), idRef_top.iphi()+1} ;
    int rangeEtatop[3] = {idRef_top.ieta()-1, idRef_top.ieta(), idRef_top.ieta()+1} ;
    int rangePhibot[3] = {idRef_bot.iphi()-1, idRef_bot.iphi(), idRef_bot.iphi()+1} ;
    int rangeEtabot[3] = {idRef_bot.ieta()-1, idRef_bot.ieta(), idRef_bot.ieta()+1} ;
    for (int i=0 ; i<3 ; i++) {
      if (rangePhitop[i] <= 0) rangePhitop[i] += 360 ;
      if (rangePhitop[i] > 360) rangePhitop[i] -= 360 ;
      if (rangeEtatop[i] <= 0 || rangeEtatop[i]>85)  rangeEtatop[i] = 999999 ;
      if (rangePhibot[i] <= 0) rangePhibot[i] += 360 ;
      if (rangePhibot[i] > 360) rangePhibot[i] -= 360 ;
      if (rangeEtabot[i] <= 0 || rangeEtabot[i]>85)  rangeEtabot[i] = 999999 ;
    }

    Etop_ = 0. ;
    Ebot_ = 0. ;
    Ntop_ = 0 ;
    Nbot_ = 0 ;
    for (unsigned int i=0;i<digi.product()->size();i++) {
      const EBDataFrame & df = (*(digi.product()))[i];
      const EBDetId & id=df.id();
      int adc ;
      float E_xtal = 0. ; 
      float mean = 0.5*(df[0].adc()+df[1].adc()) ; 
      float max = -999 ; 
      for (int samp = 0 ; samp<10 ; samp++) {
	adc = df[samp].adc() ;
	if (adc>max) max = adc ;
      }
      E_xtal = (adc-mean) ;

      for (int phiIndex=0 ; phiIndex<3 ; phiIndex++)
	for (int etaIndex = 0 ; etaIndex<3 ; etaIndex++) {
	  if (id.iphi() == rangePhitop[phiIndex] && id.ieta() == rangeEtatop[etaIndex]) {
	    Etop_ += E_xtal ;
	    Ntop_ ++ ;
	  }
	  if (id.iphi() == rangePhibot[phiIndex] && id.ieta() == rangeEtabot[etaIndex]) {
	    Ebot_ += E_xtal ;
	    Nbot_ ++ ;
	  }
	}
    }

    iphitop_ = idRef_top.iphi() ;
    ietatop_ = idRef_top.ieta() ;
    iphibot_ = idRef_bot.iphi() ;
    ietabot_ = idRef_bot.ieta() ;
    treetopbot_->Fill() ;
  }

}
  
