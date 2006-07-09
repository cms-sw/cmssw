#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "DataFormats/Common/interface/ProductID.h"
#include <memory>
#include <vector>

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

  /** Sanity check on the DCC FIR filter weights. Log warning or
   * error message if an unexpected weight set is found. In principle
   * it is checked that the maximum weight is applied to the expected
   * maximum sample.
   */
  void
  checkWeights(const edm::Event& evt, const edm::ProductID& noZSDigiId) const;

  /** Gets the value of the digitizer binOfMaximum parameter.
   * @param noZsDigiId product ID of the non-suppressed digis
   * @param binOfMax [out] set the parameter value if found
   * @return true on success, false otherwise
   */
  bool
  getBinOfMax(const edm::Event& evt, const edm::ProductID& noZsDigiId,
	      int& binOfMax) const;
  
  const EBDigiCollection*
  getEBDigis(edm::Event& event) const;

  const EEDigiCollection*
  getEEDigis(edm::Event& event) const;

  const EcalTrigPrimDigiCollection*
  getTrigPrims(edm::Event& event) const;
  
  /// call these once an event, to make sure everything
  /// is up-to-date
  void
  checkGeometry(const edm::EventSetup & eventSetup);
  void
  checkTriggerMap(const edm::EventSetup & eventSetup);

  void
  printTTFlags(const EcalTrigPrimDigiCollection& tp, std::ostream& os) const;
  
  void
  printSRFHeader(std::ostream& os);
  
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
  edm::ParameterSet params_;

  bool trigPrimBypass_;

  /** Number of event whose TT and SR flags must be dumped into a file.
   */
  int dumpFlags_;

};

#endif 
