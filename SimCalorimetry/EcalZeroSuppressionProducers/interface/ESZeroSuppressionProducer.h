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
  const std::string digiProducer_;
  const std::string ESdigiCollection_;
  const std::string ESZSdigiCollection_;

  const edm::EDGetTokenT<ESDigiCollection> ES_token;
  const edm::ESGetToken<ESThresholds, ESThresholdsRcd> esthresholdsToken_;
  const edm::ESGetToken<ESPedestals, ESPedestalsRcd> espedsToken_;
};

#endif
