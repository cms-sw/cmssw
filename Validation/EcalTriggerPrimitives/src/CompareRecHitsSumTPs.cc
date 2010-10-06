// -*- C++ -*-
//
// Package:    CompareRecHitsSumTPs
// Class:      CompareRecHitsSumTPs
// 
/**\class CompareRecHitsSumTPs CompareRecHitsSumTPs.cc Validation/CompareRecHitsSumTPs/src/CompareRecHitsSumTPs.cc

 Description: Get the eRecHit 

 Implementation:
     Save EventId info, eRecHit and TTId into a tree
     Save eRecHit into histograms
*/
//
// Original Author:  Emilia Lubenova Becheva,40 4-A24,+41227678742,
//         Created:  Mon Sep 13 14:24:36 CEST 2010
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/EcalEndcapGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "TH1I.h"
#include "TH2F.h"
#include "TFile.h"
#include "TTree.h"

#include <map>
#include <vector>

//
// class declaration
//

typedef unsigned int EventNumber_t;
typedef unsigned int LuminosityBlockNumber_t;
typedef unsigned int RunNumber_t;

class CompareRecHitsSumTPs : public edm::EDAnalyzer {
   public:
      explicit CompareRecHitsSumTPs(const edm::ParameterSet&);
      ~CompareRecHitsSumTPs();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      
	
   private:
      const CaloSubdetectorGeometry * theEndcapGeometry_ ;
      const CaloSubdetectorGeometry * theBarrelGeometry_ ;
      edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;

      edm::InputTag EcalRecHitCollectionEB_;
      edm::InputTag EcalRecHitCollectionEE_;
      
      double towerEnergy;
      TH1I *ecalRecHit_[2]; 
      TH1I *ecalRecHitADC_[2];     
      TFile *histFile_;
      TTree *t_;
      
      
      // data for tree
      RunNumber_t runNbRecHit;
      LuminosityBlockNumber_t lumiBlockRecHit;
      LuminosityBlockNumber_t eventNbRecHit;
      
      std::vector<unsigned int> towIdEBRecHit;
      std::vector<unsigned int> towIdEERecHit;
      std::vector<double> ecalRecHitADC_EB;
      std::vector<double> ecalRecHitADC_EE;
      
      std::vector<std::string> ecal_parts_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

CompareRecHitsSumTPs::CompareRecHitsSumTPs(const edm::ParameterSet& iConfig)
{  
  //now do what ever initialization is needed
  EcalRecHitCollectionEB_ = iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEB");
  EcalRecHitCollectionEE_ = iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEE");

  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  histFile_=new TFile("/tmp/ebecheva/histosRecHits.root","RECREATE");

  // Create histos for EB and EE eRecHit in ADC units and in GeV 
  for (unsigned int i=0;i<2;++i) {
    // Energy
    char t[30];
    char tADC[30];
    sprintf(t,"%s_eRecHit",ecal_parts_[i].c_str());  
    sprintf(tADC,"%s_eRecHitADC",ecal_parts_[i].c_str());
        
    ecalRecHitADC_[i]=new TH1I(tADC,"Et",255,0,255);
    ecalRecHit_[i]=new TH1I(t,"Et",800,0,100);
       
  }
  
  
  // Tree containing the eventID and the eRecHit
  t_ = new TTree("TreeERecHit","ERecHit");
  t_->Branch("runNbRecHit",&runNbRecHit,"runNbRecHit/i");
  t_->Branch("lumiBlockRecHit",&lumiBlockRecHit,"lumiBlockRecHit/i");
  t_->Branch("eventNbRecHit",&eventNbRecHit,"eventNbRecHit/i");
  t_->Branch("towIdEBRecHit",&towIdEBRecHit);
  t_->Branch("towIdEERecHit",&towIdEERecHit);
  t_->Branch("ecalRecHitADC_EB",&ecalRecHitADC_EB);
  t_->Branch("ecalRecHitADC_EE",&ecalRecHitADC_EE);
}


CompareRecHitsSumTPs::~CompareRecHitsSumTPs()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   histFile_->Write();
   histFile_->Close();
   
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CompareRecHitsSumTPs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
    
  runNbRecHit = iEvent.id().run();
  lumiBlockRecHit = iEvent.id().luminosityBlock();
  eventNbRecHit = iEvent.id().event();
  
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  iSetup.get<EcalEndcapGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  iSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);

  iSetup.get<IdealGeometryRecord>().get(eTTmap_);

  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);


    ///////////////////////////
    // Get rechits and spikes
    ///////////////////////////

    // channel status
    edm::ESHandle<EcalChannelStatus> pChannelStatus;
    iSetup.get<EcalChannelStatusRcd>().get(pChannelStatus);
    const EcalChannelStatus *chStatus = pChannelStatus.product();
    
    std::map<unsigned int, double> mapTowerEB;
    std::map<unsigned int, double>::iterator itTTEB;

    std::map<unsigned int, double> mapTowerEE;
    std::map<unsigned int, double>::iterator itTTEE;

    // Get EB rechits
    edm::Handle<EcalRecHitCollection> rechitsEB; 
    
    bool got = iEvent.getByLabel(EcalRecHitCollectionEB_, rechitsEB);
    
    if (iEvent.getByLabel(EcalRecHitCollectionEB_, rechitsEB) ) {
	
      // Fill the EB map with the TT rawId	
      double ebtE0=0;
      for ( EcalRecHitCollection::const_iterator rechitItr = rechitsEB->begin(); rechitItr != rechitsEB->end(); ++rechitItr ) 
      {
        EBDetId id = rechitItr->id();
	const EcalTrigTowerDetId towid = id.tower();
	mapTowerEB[towid.rawId()]=ebtE0;
      }
      
      
      for ( EcalRecHitCollection::const_iterator rechitItr = rechitsEB->begin(); rechitItr != rechitsEB->end(); ++rechitItr ) 
      { 
	EBDetId id = rechitItr->id();
	const EcalTrigTowerDetId towid = id.tower();
		
	itTTEB = mapTowerEB.find(towid.rawId()) ;
	if (itTTEB != mapTowerEB.end()) {
	  double theta = theBarrelGeometry_->getGeometry(id)->getPosition().theta() ; 
	  (itTTEB->second) += rechitItr->energy()*sin(theta);
	}
       }
    }
    
    
    
    // Get EE rechits
    edm::Handle<EcalRecHitCollection> rechitsEE; 
    if ( iEvent.getByLabel(EcalRecHitCollectionEE_, rechitsEE) ) {
      
      // Fill the EE map with the TT rawId
      double ebtE0=0;
      for ( EcalRecHitCollection::const_iterator rechitItr = rechitsEE->begin(); rechitItr != rechitsEE->end(); ++rechitItr ) {	
      	EEDetId id = rechitItr->id();
	const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
        mapTowerEE[towid.rawId()]=ebtE0;
      }
      
      
      for ( EcalRecHitCollection::const_iterator rechitItr = rechitsEE->begin(); rechitItr != rechitsEE->end(); ++rechitItr ) {	
	EEDetId id = rechitItr->id();
	const EcalTrigTowerDetId towid = (*eTTmap_).towerOf(id);
	
 	itTTEE = mapTowerEE.find(towid.rawId());
	if (itTTEE != mapTowerEE.end()) {
	  double theta = theEndcapGeometry_->getGeometry(id)->getPosition().theta();
	  (itTTEE->second) += rechitItr->energy()*sin(theta);
	}
      }
    }
    
           
    //--------------------------- EB map --------------------------------------
    // Fill histograms and tree
    for (itTTEB = mapTowerEB.begin() ; itTTEB != mapTowerEB.end() ; ++itTTEB) {     
          ecalRecHit_[0]->Fill(itTTEB->second);
	  //Transform GeV to ADC units
	  ecalRecHitADC_[0]->Fill(itTTEB->second/0.25);
	  towIdEBRecHit.push_back(itTTEB->first);
	  ecalRecHitADC_EB.push_back(itTTEB->second/0.25);
    }

    
    //---------------------------- EE map --------------------------------------
    
    for (itTTEE = mapTowerEE.begin() ; itTTEE != mapTowerEE.end() ; ++itTTEE) {   
          ecalRecHit_[1]->Fill(itTTEE->second);
	  //Transform GeV to ADC units
	  ecalRecHitADC_[1]->Fill(itTTEE->second/0.25);
	  towIdEERecHit.push_back(itTTEE->first);
	  ecalRecHitADC_EE.push_back(itTTEE->second/0.25);	            
    }
        
    t_->Fill();
   
}


// ------------ method called once each job just before starting event loop  ------------
void 
CompareRecHitsSumTPs::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CompareRecHitsSumTPs::endJob() {
 
   for (unsigned int i=0;i<2;++i) {
     ecalRecHit_[i]->Write();
     ecalRecHitADC_[i]->Write();   
   }
   
   
}

//define this as a plug-in
DEFINE_FWK_MODULE(CompareRecHitsSumTPs);
