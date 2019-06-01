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
#include "FWCore/Framework/interface/ESHandle.h"
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
  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalZeroSuppressionProducer(const edm::ParameterSet &params);
  ~EcalZeroSuppressionProducer() override;

  /**Produces the EDM products,*/
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

  void initCalibrations(const edm::EventSetup &eventSetup);

private:
  double glbBarrelThreshold_;
  double glbEndcapThreshold_;

  std::string digiProducer_;        // name of module/plugin/producer making digis
  std::string EBdigiCollection_;    // secondary name given to collection of digis
  std::string EEdigiCollection_;    // secondary name given to collection of digis
  std::string EBZSdigiCollection_;  // secondary name given to collection of digis
  std::string EEZSdigiCollection_;  // secondary name given to collection of digis

  EcalZeroSuppressor<EBDataFrame> theBarrelZeroSuppressor_;
  EcalZeroSuppressor<EEDataFrame> theEndcapZeroSuppressor_;

  edm::EDGetTokenT<EBDigiCollection> EB_token;
  edm::EDGetTokenT<EEDigiCollection> EE_token;
};

#endif
