#ifndef EcalSimAlgos_EcalSignalGenerator_h
#define EcalSimAlgos_EcalSignalGenerator_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalBaseSignalGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
// 
// This code is only to add pileup
// 

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



/** Converts digis back into analog signals, to be used
 *  as noise 
 */

#include <iostream>
#include <memory>

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
                      const double peToAEndcap)
      : EcalBaseSignalGenerator(), theEvent(nullptr), theEventPrincipal(nullptr), theInputTag(inputTag), tok_(t) {
    EcalMGPAGainRatio* defaultRatios = new EcalMGPAGainRatio();
    theDefaultGains[2] = defaultRatios->gain6Over1();
    theDefaultGains[1] = theDefaultGains[2] * (defaultRatios->gain12Over6());
    m_EBs25notCont = EBs25notCont;
    m_EEs25notCont = EEs25notCont;
    m_peToABarrel = peToABarrel;
    m_peToAEndcap = peToAEndcap;
    
    
//     std::cout << " ---> EcalSignalGenerator() : EcalBaseSignalGenerator() " << std::endl;
    
  }

  ~EcalSignalGenerator() override {}

  void initializeEvent(const edm::Event* event, const edm::EventSetup* eventSetup) {
    
//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() " << std::endl;
    
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

 
    //----
    // Ecal LaserCorrection Constants for laser correction ratio
    edm::ESHandle<EcalLaserDbService> laser;
    eventSetup->get<EcalLaserDbRecord>().get(laser);
    
//     const edm::TimeValue_t eventTimeValue = theEvent->time().value();
    const edm::TimeValue_t eventTimeValue = theEvent->run(); 
    //---- NB: this is a trick. Since the time dependent MC 
    //         will be based on "run" (and lumisection)
    //         to identify the IOV.
    //         The "time" defined here as "run" 
    //         will have to match in the generation of the tag 
    //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
    //
    m_iTime = eventTimeValue;
    
    
    
    m_lasercals = laser.product();
//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() :: eventTimeValue = " << eventTimeValue << std::endl;

    edm::ESHandle<EcalLaserDbService> laser_prime;
    eventSetup->get<EcalLaserDbRecord>().get(laser_prime);
    //     const edm::TimeValue_t eventTimeValue = event.time().value();
    m_lasercals_prime = laser_prime.product();
    
    //clear the laser cache for each event time
//     CalibCache().swap(m_valueLCCache_LC);   -> strange way to clear a collection ... why was it like this?
//     http://www.cplusplus.com/reference/unordered_map/unordered_map/clear/
//     http://www.cplusplus.com/reference/unordered_map/unordered_map/swap/
    m_valueLCCache_LC.clear();
    //----

    
    eventSetup->get<EcalIntercalibConstantsMCRcd>().get(pIcal);
    ical = pIcal.product();
    
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
    
//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() [end] " << std::endl;
    
  }

  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal* eventPrincipal, const edm::EventSetup* eventSetup) {

//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() [second version]" << std::endl;

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

    //----
    // Ecal LaserCorrection Constants for laser correction ratio
    edm::ESHandle<EcalLaserDbService> laser;
    eventSetup->get<EcalLaserDbRecord>().get(laser);
    edm::TimeValue_t eventTimeValue;
    if (theEventPrincipal) {
//       eventTimeValue = theEventPrincipal->time().value();
      eventTimeValue = theEventPrincipal->run();
      //---- NB: this is a trick. Since the time dependent MC 
      //         will be based on "run" (and lumisection)
      //         to identify the IOV.
      //         The "time" defined here as "run" 
      //         will have to match in the generation of the tag 
      //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
      //
    }
    else {
      std::cout << " theEventPrincipal not defined??? " << std::endl;
    }
    m_iTime = eventTimeValue;
    m_lasercals = laser.product();
//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() :: eventTimeValue = " << eventTimeValue << std::endl;
    
    edm::ESHandle<EcalLaserDbService> laser_prime;
    eventSetup->get<EcalLaserDbRecord>().get(laser_prime);
    //     const edm::TimeValue_t eventTimeValue = event.time().value();
    m_lasercals_prime = laser_prime.product();
    
    //clear the laser cache for each event time
    //     CalibCache().swap(m_valueLCCache_LC);   -> strange way to clear a collection ... why was it like this?
    //     http://www.cplusplus.com/reference/unordered_map/unordered_map/clear/
    //     http://www.cplusplus.com/reference/unordered_map/unordered_map/swap/
    m_valueLCCache_LC.clear();
    //----
    
    
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
    
//     std::cout << " ---> EcalSignalGenerator() : initializeEvent() [end] " << std::endl;
    
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
    
//     return 1.0;
    
//     const edm::TimeValue_t m_iTime = event.time().value();
    const edm::Timestamp& evtTimeStamp = edm::Timestamp(m_iTime);
    
//     std::cout << " findLaserConstant_LC::evtTimeStamp = " << evtTimeStamp << std::endl;
    
    return (m_lasercals->getLaserCorrection(detId, evtTimeStamp));
    
  }
  
  
  //---- LC at the beginning of the time (first IOV of the GT == first time)
  double findLaserConstant_LC_prime(const DetId& detId) const {
    
//     return 1.0;
    
    int temp_iTime = 0; //---- Correct to set the time to 0 --> the "LC'" is the first IOV of the tag MC to be used
    const edm::Timestamp& evtTimeStamp = edm::Timestamp(temp_iTime);
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
