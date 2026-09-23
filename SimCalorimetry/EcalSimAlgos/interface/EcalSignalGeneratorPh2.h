#ifndef SimCalorimetry_EcalSimAlgos_EcalSignalGeneratorPh2_h
#define SimCalorimetry_EcalSimAlgos_EcalSignalGeneratorPh2_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalCATIAGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalCATIAGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalBaseSignalGenerator.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"

// needed for LC'/LC correction for time dependent MC
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecordMC.h"

/** Converts digis back into analog signals, to be used
 *  as noise 
 */

namespace edm {
  class ModuleCallingContext;
}

template <class ECALDIGITIZERTRAITS>
class EcalSignalGeneratorPh2 : public EcalBaseSignalGenerator {
public:
  typedef typename ECALDIGITIZERTRAITS::Digi DIGI;
  typedef typename ECALDIGITIZERTRAITS::DigiCollection COLLECTION;

  typedef std::unordered_map<uint32_t, double> CalibCache;

  EcalSignalGeneratorPh2() : EcalBaseSignalGenerator() {}
  EcalSignalGeneratorPh2(const EcalSignalGeneratorPh2&) = delete;
  EcalSignalGeneratorPh2& operator=(const EcalSignalGeneratorPh2&) = delete;

  EcalSignalGeneratorPh2(edm::ConsumesCollector& cc,
                         const edm::InputTag& inputTag,
                         const double ebs25notCont,
                         const double peToA,
                         const bool timeDependent = false);

  ~EcalSignalGeneratorPh2() override = default;

  void initializeEvent(const edm::Event* event, const edm::EventSetup* eventSetup);
  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal* eventPrincipal, const edm::EventSetup* eventSetup);

  virtual void fill(edm::ModuleCallingContext const* mcc);

private:
  inline bool validDigi(const DIGI& digi) {
    for (int id = 0; id < digi.size(); ++id) {
      if (digi[id].adc() > 0) {
        return true;
      }
    }
    return false;
  }

  void fillNoiseSignals() override {}
  void fillNoiseSignals(CLHEP::HepRandomEngine*) override {}

  CaloSamples samplesInPE(const DIGI& digi);

  const edm::ESGetToken<EcalCATIAGainRatios, EcalCATIAGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalIntercalibConstantsMC, EcalIntercalibConstantsMCRcd> interCalibConstantsMCToken_;
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVConstantToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserDbToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecordMC> laserDbMCToken_;

  /// these fields are set in initializeEvent()
  const edm::Event* theEvent_;
  const edm::EventPrincipal* theEventPrincipal_;

  const EcalCATIAGainRatios* gainRatios_;

  /// these come from the ParameterSet
  const edm::InputTag theInputTag_;
  const edm::EDGetTokenT<COLLECTION> tok_;

  const double ebs25notCont_;
  const double peToA_;

  double maxEne_;  // max attainable energy in the ecal barrel

  const EcalIntercalibConstantsMC* ical_;

  const bool timeDependent_;
  edm::TimeValue_t iTime_;
  CalibCache valueLCCache_LC_;
  CalibCache valueLCCache_LC_prime_;
  const EcalLaserDbService* lasercals_;
  const EcalLaserDbService* lasercals_prime_;
};

typedef EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2> EBSignalGeneratorPh2;
#endif
