// -*- C++ -*-
//
// Class:      EcalTPInputAnalyzer
// 
/**\class EcalTPInputAnalyzer

 Description: test of the input of EcalTrigPrimProducer

*/
//
//
// Original Author:  Ursula Berthon
//         Created:  Thu Jul 4 11:38:38 CEST 2005
// $Id: EcalTPInputAnalyzer.cc,v 1.5 2008/01/17 13:40:52 uberthon Exp $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EcalTPInputAnalyzer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"

using namespace edm;

EcalTPInputAnalyzer::EcalTPInputAnalyzer(const edm::ParameterSet& iConfig)

{
  histfile_=new TFile("histos.root","UPDATE");
  histEndc = new TH1I("AdcE","Adc-s for Endcap",100,0.,5000.);
  histBar = new TH1I("AdcB","Adc-s for Barrel",100,0.,5000.);
  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

//   for (unsigned int i=0;i<2;++i) {
//     ecal_et_[i]=new TH1I(ecal_parts_[i].c_str(),"Et",255,0,255);
//     char title[30];
//     sprintf(title,"%s_ttf",ecal_parts_[i].c_str());
//     ecal_tt_[i]=new TH1I(title,"TTF",10,0,10);
//     sprintf(title,"%s_fgvb",ecal_parts_[i].c_str());
//     ecal_fgvb_[i]=new TH1I(title,"FGVB",10,0,10);
//   }
   producer_= iConfig.getParameter<std::string>("Producer");
   ebLabel_= iConfig.getParameter<std::string>("EBLabel");
   eeLabel_= iConfig.getParameter<std::string>("EELabel");
}


EcalTPInputAnalyzer::~EcalTPInputAnalyzer()
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
EcalTPInputAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  bool barrel=true;
  edm::Handle<EBDigiCollection> ebDigis;
  edm::Handle<EEDigiCollection> eeDigis;
  if (!iEvent.getByLabel(producer_,ebLabel_,ebDigis)) {
    barrel=false;
    edm::LogWarning("EcalTPG") <<" Couldnt find Barrel dataframes with Producer:"<<producer_<<" and label: "<<ebLabel_;
  }
  bool endcap=true;
  if (!iEvent.getByLabel(producer_,eeLabel_,eeDigis)) {
    endcap=false;
    edm::LogWarning("EcalTPG") <<" Couldnt find Endcap dataframes with Producer:"<<producer_<<" and label: "<<eeLabel_;
  }
  //barrel
  if (barrel) {
    const EBDigiCollection *ebdb=ebDigis.product();
    for (unsigned int i=0;i<ebDigis->size();++i) {
      EBDataFrame ebdf=(*ebdb)[i];
      int nrSamples=ebdf.size();
      //unsigned int nrSamples=(ebDigis.product())[i].size();
      for (int is=0;is<nrSamples;++is) {
	//	EcalMGPASample sam=((ebDigis.product())[i])[is];
	EcalMGPASample sam=ebdf[is];
	histBar->Fill(sam.adc());
      }
    }
  }
  //endcap
  if (endcap) {
    const EEDigiCollection *eedb=eeDigis.product();
    for (unsigned int i=0;i<eeDigis->size();++i) {
      EEDataFrame eedf=(*eedb)[i];
      int nrSamples=eedf.size();
      for (int is=0;is<nrSamples;++is) {
	EcalMGPASample sam=eedf[is];
	histEndc->Fill(sam.adc());
      }
    }
  }
//   // Get input
//   edm::Handle<EcalTrigPrimDigiCollection> tp;
//   iEvent.getByLabel(label_,producer_,tp);
//   for (unsigned int i=0;i<tp.product()->size();i++) {
//     EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
//     int subdet=d.id().subDet()-1;
//       ecal_et_[subdet]->Fill(d.compressedEt());
//       ecal_tt_[subdet]->Fill(d.ttFlag());
//       ecal_fgvb_[subdet]->Fill(d.fineGrain());
//   }
}

void
EcalTPInputAnalyzer::endJob(){
  histEndc ->Write();
  histBar  ->Write();
}
  
    


