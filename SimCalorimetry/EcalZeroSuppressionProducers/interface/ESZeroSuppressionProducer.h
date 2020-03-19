#ifndef ESZEROSUPPRESSIONPRODUCER_H
#define ESZEROSUPPRESSIONPRODUCER_H

#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/ESThresholdsRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ESZeroSuppressionProducer : public edm::stream::EDProducer<> {
public:
  explicit ESZeroSuppressionProducer(const edm::ParameterSet &ps);
  ~ESZeroSuppressionProducer() override;

  /**Produces the EDM products,*/
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  std::string digiProducer_;
  std::string ESdigiCollection_;
  std::string ESZSdigiCollection_;

  edm::ESHandle<ESThresholds> esthresholds_;
  edm::ESHandle<ESPedestals> espeds_;

  edm::EDGetTokenT<ESDigiCollection> ES_token;
};

#endif
