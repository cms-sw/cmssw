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
//
//

// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "EcalTPInputAnalyzer.h"

EcalTPInputAnalyzer::EcalTPInputAnalyzer(const edm::ParameterSet &iConfig)
    : producer_(iConfig.getParameter<std::string>("Producer")),
      ebLabel_(iConfig.getParameter<std::string>("EBLabel")),
      eeLabel_(iConfig.getParameter<std::string>("EELabel")),
      ebToken_(consumes<EBDigiCollection>(edm::InputTag(producer_, ebLabel_))),
      eeToken_(consumes<EEDigiCollection>(edm::InputTag(producer_, eeLabel_))) {
  usesResource(TFileService::kSharedResource);

  edm::Service<TFileService> fs;
  histEndc = fs->make<TH1I>("AdcE", "Adc-s for Endcap", 100, 0., 5000.);
  histBar = fs->make<TH1I>("AdcB", "Adc-s for Barrel", 100, 0., 5000.);
  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  //   for (unsigned int i=0;i<2;++i) {
  //     ecal_et_[i] = fs->make<TH1I>(ecal_parts_[i].c_str(),"Et",255,0,255);
  //     char title[30];
  //     sprintf(title,"%s_ttf",ecal_parts_[i].c_str());
  //     ecal_tt_[i] = fs->make<TH1I>(title,"TTF",10,0,10);
  //     sprintf(title,"%s_fgvb",ecal_parts_[i].c_str());
  //     ecal_fgvb_[i] = fs->make<TH1I>(title,"FGVB",10,0,10);
  //   }
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void EcalTPInputAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  bool barrel = true;
  const edm::Handle<EBDigiCollection> &ebDigis = iEvent.getHandle(ebToken_);
  if (!ebDigis.isValid()) {
    barrel = false;
    edm::LogWarning("EcalTPG") << " Couldnt find Barrel dataframes with Producer:" << producer_
                               << " and label: " << ebLabel_;
  }
  bool endcap = true;
  const edm::Handle<EEDigiCollection> &eeDigis = iEvent.getHandle(eeToken_);
  if (!eeDigis.isValid()) {
    endcap = false;
    edm::LogWarning("EcalTPG") << " Couldnt find Endcap dataframes with Producer:" << producer_
                               << " and label: " << eeLabel_;
  }
  // barrel
  if (barrel) {
    const EBDigiCollection *ebdb = ebDigis.product();
    for (unsigned int i = 0; i < ebDigis->size(); ++i) {
      EBDataFrame ebdf = (*ebdb)[i];
      int nrSamples = ebdf.size();
      // unsigned int nrSamples=(ebDigis.product())[i].size();
      for (int is = 0; is < nrSamples; ++is) {
        //	EcalMGPASample sam=((ebDigis.product())[i])[is];
        EcalMGPASample sam = ebdf[is];
        histBar->Fill(sam.adc());
      }
    }
  }
  // endcap
  if (endcap) {
    const EEDigiCollection *eedb = eeDigis.product();
    for (unsigned int i = 0; i < eeDigis->size(); ++i) {
      EEDataFrame eedf = (*eedb)[i];
      int nrSamples = eedf.size();
      for (int is = 0; is < nrSamples; ++is) {
        EcalMGPASample sam = eedf[is];
        histEndc->Fill(sam.adc());
      }
    }
  }
  //   // Get input
  //   const edm::Handle<EcalTrigPrimDigiCollection>& tp = iEvent.getHandle(tpToken_);
  //   for (unsigned int i=0;i<tp.product()->size();i++) {
  //     EcalTriggerPrimitiveDigi d=(*(tp.product()))[i];
  //     int subdet=d.id().subDet()-1;
  //       ecal_et_[subdet]->Fill(d.compressedEt());
  //       ecal_tt_[subdet]->Fill(d.ttFlag());
  //       ecal_fgvb_[subdet]->Fill(d.fineGrain());
  //   }
}
