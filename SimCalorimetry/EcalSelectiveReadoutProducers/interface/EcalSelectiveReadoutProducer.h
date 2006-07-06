#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
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
  
  /// call these once an event, to make sure everything
  /// is up-to-date
  void checkGeometry(const edm::EventSetup & eventSetup);
  void checkTriggerMap(const edm::EventSetup & eventSetup);

  void printTTFlags(const EcalTrigPrimDigiCollection& tp, std::ostream& os);
  
  void printSRFHeader(std::ostream& os);
  
private:
  std::auto_ptr<EcalSelectiveReadoutSuppressor> suppressor_;
  std::string digiProducer_; // name of module/plugin/producer making digis
  std::string ebdigiCollection_; // secondary name given to collection of digis
  std::string eedigiCollection_; // secondary name given to collection of digis
  std::string ebSRPdigiCollection_; // secondary name given to collection of digis
  std::string eeSRPdigiCollection_; // secondary name given to collection of digis
  std::string trigPrimProducer_; // name of module/plugin/producer making triggere primitives

  // store the pointer, so we don't have to update it every event
  const CaloGeometry * theGeometry;
  const EcalTrigTowerConstituentsMap * theTriggerTowerMap;
};

#endif 
