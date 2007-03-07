#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
 

#include "SimCalorimetry/EcalZeroSuppressionAlgos/interface/EcalZeroSuppressor.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"

class EcalZeroSuppressionProducer : public edm::EDProducer
{
public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalZeroSuppressionProducer(const edm::ParameterSet& params);
  virtual ~EcalZeroSuppressionProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

  void initCalibrations(const edm::EventSetup & eventSetup);

private:

  double glbBarrelThreshold_;
  double glbEndcapThreshold_;
  
  std::string digiProducer_; // name of module/plugin/producer making digis
  std::string EBdigiCollection_; // secondary name given to collection of digis
  std::string EEdigiCollection_; // secondary name given to collection of digis
  std::string EBZSdigiCollection_; // secondary name given to collection of digis
  std::string EEZSdigiCollection_; // secondary name given to collection of digis

  EcalZeroSuppressor<EBDataFrame> theBarrelZeroSuppressor_;
  EcalZeroSuppressor<EEDataFrame> theEndcapZeroSuppressor_;
 
};

#endif 
