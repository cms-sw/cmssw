// -*- C++ -*-
//
// Package:    TPGCheck
// Class:      TPGCheck
// 
/**\class TPGCheck TPGCheck.cc Validation/TPGSimulation/src/TPGCheck.cc

 Description: Validation of the Ecal Trigger Primitives

 Implementation:
     Save into histograms the TPs: Et, TTF and FGVB
     Save into a tree the Et and the corresponding TT rowId
     while the eventId: runNb, lumiBlock and eventNb
*/
//
// Original Author:  Muriel Cerutti
//         Created:  Thu Oct 26 10:47:17 CEST 2006
// $Id: TPGCheck.cc,v 1.4 2010/09/20 15:30:22 ebecheva Exp $
// Create a tree containing the TT rowId and the Et, 
// it will be used for comparison with the RecHit energy


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <vector>
#include <string>

#include "TH1I.h"
#include "TFile.h"
#include "TTree.h"

using namespace edm;
using namespace std;

//
// class declaration
//

class TPGCheck : public edm::EDAnalyzer {
   public:
      explicit TPGCheck(const edm::ParameterSet&);
      ~TPGCheck();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      TH1I *ecal_et_[2];
      TH1I *ecal_tt_[2];
      TH1I * ecal_fgvb_[2];
      
      TFile *histFile_;
      std::string label_;
      std::string producer_;
      std::vector<std::string> ecal_parts_;
      
      // data for tree
      TTree *t_;
      RunNumber_t runNbTP;
      LuminosityBlockNumber_t lumiBlockTP;
      LuminosityBlockNumber_t eventNbTP;
      
      std::vector<unsigned int> towIdEBTP;
      std::vector<unsigned int> towIdEETP;
      std::vector<double> eTPADC_EB;
      std::vector<double> eTPADC_EE;

};

//
// constructors and destructor
//
TPGCheck::TPGCheck(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   
  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  histFile_=new TFile("/tmp/ebecheva/histos.root","RECREATE");
  for (unsigned int i=0;i<2;++i) {
    // Energy
    char t[30];
    sprintf(t,"%s_energy",ecal_parts_[i].c_str());  
    ecal_et_[i]=new TH1I(t,"Et",255,0,255);
    
    // Trigger Tower flag
    char titleTTF[30];
    sprintf(titleTTF,"%s_ttf",ecal_parts_[i].c_str());
    ecal_tt_[i]=new TH1I(titleTTF,"TTF",10,0,10);
    
    // Fain Grain
    char titleFG[30];
    sprintf(titleFG,"%s_fgvb",ecal_parts_[i].c_str());
    ecal_fgvb_[i]=new TH1I(titleFG,"FGVB",10,0,10);
  }
  
   // Tree containing the eventID and the eRecHit
  t_ = new TTree("TreeETP","ETP");
  t_->Branch("runNbTP",&runNbTP,"runNbTP/i");
  t_->Branch("lumiBlockTP",&lumiBlockTP,"lumiBlockTP/i");
  t_->Branch("eventNbTP",&eventNbTP,"eventNbTP/i");
  t_->Branch("towIdEBTP",&towIdEBTP);
  t_->Branch("towIdEETP",&towIdEETP);
  t_->Branch("eTPADC_EB",&eTPADC_EB);
  t_->Branch("eTPADC_EE",&eTPADC_EE);
 
 
  label_= iConfig.getParameter<std::string>("Label");
  producer_= iConfig.getParameter<std::string>("Producer");
  
}


TPGCheck::~TPGCheck()
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
TPGCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  runNbTP = iEvent.id().run();
  lumiBlockTP = iEvent.id().luminosityBlock();
  eventNbTP = iEvent.id().event();

  // Get input
  edm::Handle<EcalTrigPrimDigiCollection> tp;
  iEvent.getByLabel(label_,producer_,tp);
  for (unsigned int i=0;i<tp.product()->size();i++) {
        
    EcalTriggerPrimitiveDigi d=(*(tp.product()))[i]; 
    
    int subdet=d.id().subDet()-1;
    // for endcap, regroup double TP-s that are generated for the 2 interior rings
    if (subdet==0) {
      
      ecal_et_[subdet]->Fill(d.compressedEt());
            
      eTPADC_EB.push_back(d.compressedEt());
      towIdEBTP.push_back(d.id().rawId());
    }
    else {
            
      if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28) {
	if (i%2){ 
	  ecal_et_[subdet]->Fill(d.compressedEt()*2.);
          eTPADC_EE.push_back(d.compressedEt()*2.);
	   towIdEETP.push_back(d.id().rawId());
	}
      }
      else{ 
        ecal_et_[subdet]->Fill(d.compressedEt());
        eTPADC_EE.push_back(d.compressedEt());
	towIdEETP.push_back(d.id().rawId());	
      }
    }
    
    ecal_tt_[subdet]->Fill(d.ttFlag());
    ecal_fgvb_[subdet]->Fill(d.fineGrain());
  }
  
    t_->Fill();
      
}


// ------------ method called once each job just before starting event loop  ------------
void 
TPGCheck::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TPGCheck::endJob() {
   for (unsigned int i=0;i<2;++i) {
    ecal_et_[i]->Write();
    ecal_tt_[i]->Write();
    ecal_fgvb_[i]->Write();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TPGCheck);
