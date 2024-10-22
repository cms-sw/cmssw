#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "SimCalorimetry/EcalZeroSuppressionAlgos/interface/EcalZeroSuppressor.h"

class EcalZeroSuppressionProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalZeroSuppressionProducer(const edm::ParameterSet &params);
  ~EcalZeroSuppressionProducer() override;

  /**Produces the EDM products,*/
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

  void initCalibrations(const edm::EventSetup &eventSetup);

private:
  const double glbBarrelThreshold_;
  const double glbEndcapThreshold_;

  const std::string digiProducer_;        // name of module/plugin/producer making digis
  const std::string ebDigiCollection_;    // secondary name given to collection of digis
  const std::string eeDigiCollection_;    // secondary name given to collection of digis
  const std::string ebZSdigiCollection_;  // secondary name given to collection of digis
  const std::string eeZSdigiCollection_;  // secondary name given to collection of digis

  EcalZeroSuppressor<EBDataFrame> theBarrelZeroSuppressor_;
  EcalZeroSuppressor<EEDataFrame> theEndcapZeroSuppressor_;

  const edm::EDGetTokenT<EBDigiCollection> ebToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeToken_;
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalToken_;
};

#endif
