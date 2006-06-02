#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
 

#include "SimCalorimetry/EcalZeroSuppressionAlgos/interface/EcalZeroSuppressor.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"

#include <memory>

class EcalSelectiveReadoutProducer : public edm::EDProducer
{
public:

  /** Constructor
   * @param params seletive readout parameters
   */
  explicit
  EcalSelectiveReadoutProducer(const edm::ParameterSet& params);

  /** Destructor
   */
  virtual
  ~EcalSelectiveReadoutProducer();

  /** Produces the EDM products
   * @param CMS event
   * @param eventSetup event conditions
   */
  virtual void
  produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:
  const EBDigiCollection*
  getEBDigis(edm::Event& event);

  const EEDigiCollection*
  getEEDigis(edm::Event& event);

  const EcalTrigPrimDigiCollection*
  getTrigPrims(edm::Event& event);
  
private:
  std::auto_ptr<EcalSelectiveReadoutSuppressor> suppressor_;
  std::string digiProducer_; // name of module/plugin/producer making digis
  std::string trigPrimProducer_; // name of module/plugin/producer making triggere primitives
};

#endif 
