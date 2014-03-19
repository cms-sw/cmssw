#ifndef ECALZEROSUPPRESSIONPRODUCER_H
#define ECALZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

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

  /** Help function to print SR flags.
   * @param ebSrFlags the action flags of EB
   * @param eeSrFlag the action flags of EE
   * @param iEvent event number. Ignored if <0.
   * @param withHeader, if true an output description is written out as header.
   */
  static void
  printSrFlags(std::ostream& os,
	       const EBSrFlagCollection& ebSrFlags,
	       const EESrFlagCollection& eeSrFlags,
	       int iEvent = -1,
	       bool withHeader = true);


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

  ///@{
  /// call these once an event, to make sure everything
  /// is up-to-date
  void
  checkGeometry(const edm::EventSetup & eventSetup);
  void
  checkTriggerMap(const edm::EventSetup & eventSetup);
  void
  checkElecMap(const edm::EventSetup & eventSetup);

  ///@}

  ///Checks validity of selective setting object is valid to be used
  ///for MC, especially checks the number of elements in the vectors
  ///@param forEmulator if true check the restriction that applies for
  ///EcalSelectiveReadoutProducer
  ///@throw cms::Exception if the setting is not valid.
  static void checkValidity(const EcalSRSettings& settings);
  
  void
  printTTFlags(const EcalTrigPrimDigiCollection& tp, std::ostream& os) const;

private:
  std::auto_ptr<EcalSelectiveReadoutSuppressor> suppressor_;
  std::string digiProducer_; // name of module/plugin/producer making digis
  std::string ebdigiCollection_; // secondary name given to collection of input digis
  std::string eedigiCollection_; // secondary name given to collection of input digis
  std::string ebSRPdigiCollection_; // secondary name given to collection of suppressed digis
  std::string eeSRPdigiCollection_; // secondary name given to collection of suppressed digis
  std::string ebSrFlagCollection_; // secondary name given to collection of SR flag digis
  std::string eeSrFlagCollection_; // secondary name given to collection of SR flag digis
  std::string trigPrimProducer_; // name of module/plugin/producer making triggere primitives
  std::string trigPrimCollection_; // name of module/plugin/producer making triggere primitives

  // store the pointer, so we don't have to update it every event
  const CaloGeometry * theGeometry;
  const EcalTrigTowerConstituentsMap * theTriggerTowerMap;
  const EcalElectronicsMapping * theElecMap;
  edm::ParameterSet params_;

  bool trigPrimBypass_;

  int trigPrimBypassMode_;

  /** Number of event whose TT and SR flags must be dumped into a file.
   */
  int dumpFlags_;

  /** switch to write out the SrFlags collections in the event
   */
  bool writeSrFlags_;

  /** Switch for suppressed digi production If false SR flags are produced
   * but selective readout is not applied on the crystal channel digis.
   */
  bool produceDigis_;

  /** SR settings
   */
  const EcalSRSettings* settings_;

  /** Switch for retrieving SR settings from condition database instead
   * of CMSSW python configuration file.
   */
  bool useCondDb_;


  /**  Special switch to turn off SR entirely using special DB entries 
   */

  bool useFullReadout_;

  /** Used when settings_ is imported from configuration file. Just used
   * for memory management. Used settings_ to access to the object
   */
  std::auto_ptr<EcalSRSettings> settingsFromFile_;
};

#endif
