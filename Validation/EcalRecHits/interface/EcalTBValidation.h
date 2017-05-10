#ifndef ECALTBVALIDATION_H
#define ECALTBVALIDATION_H

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class EcalTBValidation : public DQMEDAnalyzer {
 public:
  explicit EcalTBValidation( const edm::ParameterSet& );
  ~EcalTBValidation();
  
  void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
 private:

  bool verbose_;

  int data_;
  int xtalInBeam_;
  //std::string rootfile_;
  std::string digiCollection_;
  std::string digiProducer_;
  std::string hitCollection_;
  std::string hitProducer_;
  std::string hodoRecInfoCollection_;
  std::string hodoRecInfoProducer_;
  std::string tdcRecInfoCollection_;
  std::string tdcRecInfoProducer_;
  std::string eventHeaderCollection_;
  std::string eventHeaderProducer_;
  // fix for consumes
  edm::EDGetTokenT<EBDigiCollection> digi_Token_;
  edm::EDGetTokenT<EBUncalibratedRecHitCollection> hit_Token_;
  edm::EDGetTokenT<EcalTBHodoscopeRecInfo> hodoRec_Token_;
  edm::EDGetTokenT<EcalTBTDCRecInfo> tdcRec_Token_;
  edm::EDGetTokenT<EcalTBEventHeader> eventHeader_Token_;
  // histos
  //TH2F *h_xib,   *h_ampltdc, *h_Shape;
  //TH1F *h_hodoX, *h_hodoY;
  //TH1F *h_e1x1, *h_e3x3,  *h_e5x5;
  //TH1F *h_e1e9, *h_e1e25, *h_e9e25;
  //TH1F *h_e1x1_center, *h_e3x3_center,  *h_e5x5_center;
  //TH2F *h_e1vsX,      *h_e1vsY;
  //TH2F *h_e1e9vsX,    *h_e1e9vsY;
  //TH2F *h_e1e25vsX,   *h_e1e25vsY;
  //TH2F *h_e9e25vsX,   *h_e9e25vsY;

  MonitorElement* meETBxib_;
  MonitorElement* meETBampltdc_;
  MonitorElement* meETBShape_;
  MonitorElement* meETBhodoX_;
  MonitorElement* meETBhodoY_;
  MonitorElement* meETBe1x1_;
  MonitorElement* meETBe3x3_;
  MonitorElement* meETBe5x5_;
  MonitorElement* meETBe1e9_;
  MonitorElement* meETBe1e25_;
  MonitorElement* meETBe9e25_;
  MonitorElement* meETBe1x1_center_;
  MonitorElement* meETBe3x3_center_;
  MonitorElement* meETBe5x5_center_;
  MonitorElement* meETBe1vsX_;
  MonitorElement* meETBe1vsY_;
  MonitorElement* meETBe1e9vsX_;
  MonitorElement* meETBe1e9vsY_;
  MonitorElement* meETBe1e25vsX_;
  MonitorElement* meETBe1e25vsY_;
  MonitorElement* meETBe9e25vsX_;
  MonitorElement* meETBe9e25vsY_;

};




#endif
