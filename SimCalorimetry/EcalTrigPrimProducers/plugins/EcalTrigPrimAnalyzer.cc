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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"

#include "EcalTrigPrimAnalyzer.h"

using namespace edm;

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
  label_= iConfig.getParameter<std::string>("Label");
  producer_= iConfig.getParameter<std::string>("Producer");
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
EcalTrigPrimAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
      else {
	if (d.id().ietaAbs()==27 || d.id().ietaAbs()==28) {
	  if (i%2) ecal_et_[subdet]->Fill(d.compressedEt()*2.);
	}
	else ecal_et_[subdet]->Fill(d.compressedEt());
      }
      ecal_tt_[subdet]->Fill(d.ttFlag());
      ecal_fgvb_[subdet]->Fill(d.fineGrain());
  }
}
    


