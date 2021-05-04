#ifndef EcalSimAlgos_EcalSignalGenerator_h
#define EcalSimAlgos_EcalSignalGenerator_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalBaseSignalGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "DataFormats/Common/interface/Handle.h"

// needed for LC'/LC correction for time dependent MC
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecordMC.h"

/** Converts digis back into analog signals, to be used
 *  as noise 
 */

#include <iostream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  class ModuleCallingContext;
}

template <class ECALDIGITIZERTRAITS>
class EcalSignalGenerator : public EcalBaseSignalGenerator {
public:
  typedef typename ECALDIGITIZERTRAITS::Digi DIGI;
  typedef typename ECALDIGITIZERTRAITS::DigiCollection COLLECTION;

  typedef std::unordered_map<uint32_t, double> CalibCache;

  EcalSignalGenerator() : EcalBaseSignalGenerator() {}

  EcalSignalGenerator(const edm::InputTag& inputTag,
                      const edm::EDGetTokenT<COLLECTION>& t,
                      const double EBs25notCont,
                      const double EEs25notCont,
                      const double peToABarrel,
                      const double peToAEndcap,
                      const bool timeDependent = false)
      : EcalBaseSignalGenerator(), theEvent(nullptr), theEventPrincipal(nullptr), theInputTag(inputTag), tok_(t) {
    EcalMGPAGainRatio* defaultRatios = new EcalMGPAGainRatio();
    theDefaultGains[2] = defaultRatios->gain6Over1();
    theDefaultGains[1] = theDefaultGains[2] * (defaultRatios->gain12Over6());
    m_EBs25notCont = EBs25notCont;
    m_EEs25notCont = EEs25notCont;
    m_peToABarrel = peToABarrel;
    m_peToAEndcap = peToAEndcap;
    m_timeDependent = timeDependent;
  }

  ~EcalSignalGenerator() override {}

  void initializeEvent(const edm::Event* event, const edm::EventSetup* eventSetup) {
    theEvent = event;
    eventSetup->get<EcalGainRatiosRcd>().get(grHandle);  // find the gains
    // Ecal Intercalibration Constants
    eventSetup->get<EcalIntercalibConstantsMCRcd>().get(pIcal);
    ical = pIcal.product();
    // adc to GeV
    eventSetup->get<EcalADCToGeVConstantRcd>().get(pAgc);
    agc = pAgc.product();

    m_maxEneEB = (agc->getEBValue()) * theDefaultGains[1] * MAXADC * m_EBs25notCont;
    m_maxEneEE = (agc->getEEValue()) * theDefaultGains[1] * MAXADC * m_EEs25notCont;

    if (m_timeDependent) {
      //----
      // Ecal LaserCorrection Constants for laser correction ratio
      edm::ESHandle<EcalLaserDbService> laser;
      eventSetup->get<EcalLaserDbRecord>().get(laser);

      //
      const edm::TimeValue_t eventTimeValue = theEvent->getRun().runAuxiliary().beginTime().value();
      //
      //         The "time" will have to match in the generation of the tag
      //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
      //
      m_iTime = eventTimeValue;

      m_lasercals = laser.product();

      //
      // the "prime" is exactly the same as the usual laser service, BUT
      // it has only 1 IOV, so that effectively you are dividing IOV_n / IOV_0
      // NB: in the creation of the tag make sure the "prime" (MC) tag is prepared properly!
      // NB again: if many IOVs also in "MC" tag, then fancy things could be perfomed ... left for the future
      //
      edm::ESHandle<EcalLaserDbService> laser_prime;
      eventSetup->get<EcalLaserDbRecordMC>().get(laser_prime);
      m_lasercals_prime = laser_prime.product();

      //clear the laser cache for each event time
      CalibCache().swap(m_valueLCCache_LC);
      CalibCache().swap(m_valueLCCache_LC_prime);  //--- also the "prime" ... yes

      //----
    }

    //ES
    eventSetup->get<ESGainRcd>().get(hesgain);
    eventSetup->get<ESMIPToGeVConstantRcd>().get(hesMIPToGeV);
    eventSetup->get<ESIntercalibConstantsRcd>().get(hesMIPs);

    esgain = hesgain.product();
    esmips = hesMIPs.product();
    esMipToGeV = hesMIPToGeV.product();
    if (1.1 > esgain->getESGain())
      ESgain = 1;
    else
      ESgain = 2;
    if (ESgain == 1)
      ESMIPToGeV = esMipToGeV->getESValueLow();
    else
      ESMIPToGeV = esMipToGeV->getESValueHigh();
  }

  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal* eventPrincipal, const edm::EventSetup* eventSetup) {
    theEventPrincipal = eventPrincipal;
    eventSetup->get<EcalGainRatiosRcd>().get(grHandle);  // find the gains
    // Ecal Intercalibration Constants
    eventSetup->get<EcalIntercalibConstantsMCRcd>().get(pIcal);
    ical = pIcal.product();
    // adc to GeV
    eventSetup->get<EcalADCToGeVConstantRcd>().get(pAgc);
    agc = pAgc.product();
    m_maxEneEB = (agc->getEBValue()) * theDefaultGains[1] * MAXADC * m_EBs25notCont;
    m_maxEneEE = (agc->getEEValue()) * theDefaultGains[1] * MAXADC * m_EEs25notCont;

    if (m_timeDependent) {
      //----
      // Ecal LaserCorrection Constants for laser correction ratio
      edm::ESHandle<EcalLaserDbService> laser;
      eventSetup->get<EcalLaserDbRecord>().get(laser);
      edm::TimeValue_t eventTimeValue = 0;
      if (theEventPrincipal) {
        //
        eventTimeValue = theEventPrincipal->runPrincipal().beginTime().value();
        //
        //         The "time" will have to match in the generation of the tag
        //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
        //
      } else {
        edm::LogError("EcalSignalGenerator") << " theEventPrincipal not defined??? " << std::endl;
      }
      m_iTime = eventTimeValue;
      m_lasercals = laser.product();

      edm::ESHandle<EcalLaserDbService> laser_prime;
      eventSetup->get<EcalLaserDbRecordMC>().get(laser_prime);
      m_lasercals_prime = laser_prime.product();

      //clear the laser cache for each event time
      CalibCache().swap(m_valueLCCache_LC);
      CalibCache().swap(m_valueLCCache_LC_prime);  //--- also the "prime" ... yes
      //----
    }

    //ES
    eventSetup->get<ESGainRcd>().get(hesgain);
    eventSetup->get<ESMIPToGeVConstantRcd>().get(hesMIPToGeV);
    eventSetup->get<ESIntercalibConstantsRcd>().get(hesMIPs);

    esgain = hesgain.product();
    esmips = hesMIPs.product();
    esMipToGeV = hesMIPToGeV.product();
    if (1.1 > esgain->getESGain())
      ESgain = 1;
    else
      ESgain = 2;
    if (ESgain == 1)
      ESMIPToGeV = esMipToGeV->getESValueLow();
    else
      ESMIPToGeV = esMipToGeV->getESValueHigh();
  }

  virtual void fill(edm::ModuleCallingContext const* mcc) {
    theNoiseSignals.clear();
    edm::Handle<COLLECTION> pDigis;
    const COLLECTION* digis = nullptr;
    // try accessing by whatever is set, Event or EventPrincipal
    if (theEvent) {
      if (theEvent->getByToken(tok_, pDigis)) {
        digis = pDigis.product();  // get a ptr to the product
      } else {
        throw cms::Exception("EcalSignalGenerator") << "Cannot find input data " << theInputTag;
      }
    } else if (theEventPrincipal) {
      std::shared_ptr<edm::Wrapper<COLLECTION> const> digisPTR =
          edm::getProductByTag<COLLECTION>(*theEventPrincipal, theInputTag, mcc);
      if (digisPTR) {
        digis = digisPTR->product();
      }
    } else {
      throw cms::Exception("EcalSignalGenerator") << "No Event or EventPrincipal was set";
    }

    if (digis) {
      // loop over digis, adding these to the existing maps
      for (typename COLLECTION::const_iterator it = digis->begin(); it != digis->end(); ++it) {
        // need to convert to something useful
        if (validDigi(*it)) {
          theNoiseSignals.push_back(samplesInPE(*it));
        }
      }
    }
    //else { std::cout << " NO digis for this input: " << theInputTag << std::endl;}
  }

private:
  bool validDigi(const DIGI& digi) {
    int DigiSum = 0;
    for (int id = 0; id < digi.size(); id++) {
      if (digi[id].adc() > 0)
        ++DigiSum;
    }
    return (DigiSum > 0);
  }

  void fillNoiseSignals() override {}
  void fillNoiseSignals(CLHEP::HepRandomEngine*) override {}

  // much of this stolen from EcalSimAlgos/EcalCoder

  enum {
    NBITS = 12,            // number of available bits
    MAXADC = 4095,         // 2^12 -1,  adc max range
    ADCGAINSWITCH = 4079,  // adc gain switch
    NGAINS = 3
  };  // number of electronic gains

  CaloSamples samplesInPE(const DIGI& digi);  // have to define this separately for ES

  //---- LC that depends with time
  double findLaserConstant_LC(const DetId& detId) const {
    const edm::Timestamp& evtTimeStamp = edm::Timestamp(m_iTime);
    return (m_lasercals->getLaserCorrection(detId, evtTimeStamp));
  }

  //---- LC at the beginning of the time (first IOV of the GT == first time)
  //---- Using the different "tag", the one with "MC": exactly the same function as findLaserConstant_LC but with a different object
  double findLaserConstant_LC_prime(const DetId& detId) const {
    const edm::Timestamp& evtTimeStamp = edm::Timestamp(m_iTime);
    return (m_lasercals_prime->getLaserCorrection(detId, evtTimeStamp));
  }

  const std::vector<float> GetGainRatios(const DetId& detid) {
    std::vector<float> gainRatios(4);
    // get gain ratios
    EcalMGPAGainRatio theRatio = (*grHandle)[detid];

    gainRatios[0] = 0.;
    gainRatios[3] = 1.;
    gainRatios[2] = theRatio.gain6Over1();
    gainRatios[1] = theRatio.gain6Over1() * theRatio.gain12Over6();

    return gainRatios;
  }

  double fullScaleEnergy(const DetId& detId) const { return detId.subdetId() == EcalBarrel ? m_maxEneEB : m_maxEneEE; }

  double peToAConversion(const DetId& detId) const {
    return detId.subdetId() == EcalBarrel ? m_peToABarrel : m_peToAEndcap;
  }

  /// these fields are set in initializeEvent()
  const edm::Event* theEvent;
  const edm::EventPrincipal* theEventPrincipal;

  edm::ESHandle<EcalGainRatios> grHandle;
  edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
  edm::ESHandle<EcalADCToGeVConstant> pAgc;

  /// these come from the ParameterSet
  edm::InputTag theInputTag;
  edm::EDGetTokenT<COLLECTION> tok_;

  edm::ESHandle<ESGain> hesgain;
  edm::ESHandle<ESMIPToGeVConstant> hesMIPToGeV;
  edm::ESHandle<ESIntercalibConstants> hesMIPs;

  const ESGain* esgain;
  const ESIntercalibConstants* esmips;
  const ESMIPToGeVConstant* esMipToGeV;
  int ESgain;
  double ESMIPToGeV;

  double m_EBs25notCont;
  double m_EEs25notCont;

  double m_peToABarrel;
  double m_peToAEndcap;

  double m_maxEneEB;  // max attainable energy in the ecal barrel
  double m_maxEneEE;  // max attainable energy in the ecal endcap

  const EcalADCToGeVConstant* agc;
  const EcalIntercalibConstantsMC* ical;

  bool m_timeDependent;
  edm::TimeValue_t m_iTime;
  CalibCache m_valueLCCache_LC;
  CalibCache m_valueLCCache_LC_prime;
  const EcalLaserDbService* m_lasercals;
  const EcalLaserDbService* m_lasercals_prime;

  double theDefaultGains[NGAINS];
};

typedef EcalSignalGenerator<EBDigitizerTraits> EBSignalGenerator;
typedef EcalSignalGenerator<EEDigitizerTraits> EESignalGenerator;
typedef EcalSignalGenerator<ESDigitizerTraits> ESSignalGenerator;

#endif
