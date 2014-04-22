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

/** Converts digis back into analog signals, to be used
 *  as noise 
 */

#include <iostream>

namespace edm {
  class ModuleCallingContext;
}

template<class ECALDIGITIZERTRAITS>
class EcalSignalGenerator : public EcalBaseSignalGenerator
{
public:
  typedef typename ECALDIGITIZERTRAITS::Digi DIGI;
  typedef typename ECALDIGITIZERTRAITS::DigiCollection COLLECTION;

  EcalSignalGenerator():EcalBaseSignalGenerator() { }

    EcalSignalGenerator(const edm::InputTag & inputTag, const edm::EDGetTokenT<COLLECTION> &t, const double EBs25notCont, const double EEs25notCont, const double peToABarrel, const double peToAEndcap)
    : EcalBaseSignalGenerator(), 
      theEvent(0), 
      theEventPrincipal(0), 
      theInputTag(inputTag), 
      tok_(t)
      { 
	EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();
	theDefaultGains[2] = defaultRatios->gain6Over1() ;
	theDefaultGains[1] = theDefaultGains[2]*(defaultRatios->gain12Over6()) ;
	m_EBs25notCont = EBs25notCont; 
	m_EEs25notCont = EEs25notCont; 
	m_peToABarrel = peToABarrel;
	m_peToAEndcap = peToAEndcap;

      }

  virtual ~EcalSignalGenerator() {}


  void initializeEvent(const edm::Event * event, const edm::EventSetup * eventSetup)
  {
    theEvent = event;
    eventSetup->get<EcalGainRatiosRcd>().get(grHandle); // find the gains
    // Ecal Intercalibration Constants
    eventSetup->get<EcalIntercalibConstantsMCRcd>().get( pIcal ) ;
    ical = pIcal.product();
    // adc to GeV
    eventSetup->get<EcalADCToGeVConstantRcd>().get(pAgc);
    agc = pAgc.product();

    m_maxEneEB = (agc->getEBValue())*theDefaultGains[1]*MAXADC*m_EBs25notCont  ;
    m_maxEneEE = (agc->getEEValue())*theDefaultGains[1]*MAXADC*m_EEs25notCont  ;

    //ES
    eventSetup->get<ESGainRcd>().               get( hesgain      ) ;
    eventSetup->get<ESMIPToGeVConstantRcd>().   get( hesMIPToGeV  ) ;
    eventSetup->get<ESIntercalibConstantsRcd>().get( hesMIPs      ) ;

    esgain     = hesgain.product()      ;
    esmips     = hesMIPs.product()      ;
    esMipToGeV = hesMIPToGeV.product()  ;
    if( 1.1 > esgain->getESGain() ) ESgain = 1;
    else ESgain = 2;
    if( ESgain ==1 ) ESMIPToGeV = esMipToGeV->getESValueLow();
    else ESMIPToGeV = esMipToGeV->getESValueHigh();
  }

  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal * eventPrincipal, const edm::EventSetup * eventSetup)
  {
    theEventPrincipal = eventPrincipal;
    eventSetup->get<EcalGainRatiosRcd>().get(grHandle);  // find the gains
    // Ecal Intercalibration Constants
    eventSetup->get<EcalIntercalibConstantsMCRcd>().get( pIcal ) ;
    ical = pIcal.product();
    // adc to GeV
    eventSetup->get<EcalADCToGeVConstantRcd>().get(pAgc);
    agc = pAgc.product();
    m_maxEneEB = (agc->getEBValue())*theDefaultGains[1]*MAXADC*m_EBs25notCont  ;
    m_maxEneEE = (agc->getEEValue())*theDefaultGains[1]*MAXADC*m_EEs25notCont  ;

    //ES
    eventSetup->get<ESGainRcd>().               get( hesgain      ) ;
    eventSetup->get<ESMIPToGeVConstantRcd>().   get( hesMIPToGeV  ) ;
    eventSetup->get<ESIntercalibConstantsRcd>().get( hesMIPs      ) ;

    esgain     = hesgain.product()      ;
    esmips     = hesMIPs.product()      ;
    esMipToGeV = hesMIPToGeV.product()  ;
    if( 1.1 > esgain->getESGain() ) ESgain = 1;
    else ESgain = 2;
    if( ESgain ==1 ) ESMIPToGeV = esMipToGeV->getESValueLow();
    else ESMIPToGeV = esMipToGeV->getESValueHigh();
  }

  virtual void fill(edm::ModuleCallingContext const* mcc)
  {

    theNoiseSignals.clear();
    edm::Handle<COLLECTION> pDigis;
    const COLLECTION *  digis = 0;
    // try accessing by whatever is set, Event or EventPrincipal
    if(theEvent) 
     {
      if( theEvent->getByToken(tok_, pDigis) ) {
        digis = pDigis.product(); // get a ptr to the product
        //LogTrace("EcalSignalGenerator") << "total # digis  for "  << theInputTag << " " <<  digis->size();
      }
      else
      {
        throw cms::Exception("EcalSignalGenerator") << "Cannot find input data " << theInputTag;
      }
    }
    else if(theEventPrincipal)
    {
       boost::shared_ptr<edm::Wrapper<COLLECTION>  const> digisPTR =
          edm::getProductByTag<COLLECTION>(*theEventPrincipal, theInputTag, mcc );
       if(digisPTR) {
          digis = digisPTR->product();
       }
    }
    else
    {
      throw cms::Exception("EcalSignalGenerator") << "No Event or EventPrincipal was set";
    }

    if (digis)
    {
      // loop over digis, adding these to the existing maps
      for(typename COLLECTION::const_iterator it  = digis->begin();
          it != digis->end(); ++it) 
      {
	// need to convert to something useful
	if(validDigi(*it)){
	  theNoiseSignals.push_back(samplesInPE(*it));
	}
      }
    }
    //else { std::cout << " NO digis for this input: " << theInputTag << std::endl;}
  }

private:

  bool validDigi(const DIGI & digi)
  {
    int DigiSum = 0;
    for(int id = 0; id<digi.size(); id++) {
      if(digi[id].adc() > 0) ++DigiSum;
    }
    return(DigiSum>0);
  }

  // much of this stolen from EcalSimAlgos/EcalCoder

  enum { NBITS         =   12 , // number of available bits
	 MAXADC        = 4095 , // 2^12 -1,  adc max range
	 ADCGAINSWITCH = 4079 , // adc gain switch
	 NGAINS        =    3 };  // number of electronic gains

  CaloSamples samplesInPE(const DIGI & digi);  // have to define this separately for ES

  const std::vector<float>  GetGainRatios(const DetId& detid) {

    std::vector<float> gainRatios(4);
    // get gain ratios  
    EcalMGPAGainRatio theRatio= (*grHandle)[detid];
    
    gainRatios[0] = 0.;
    gainRatios[3] = 1.;
    gainRatios[2] = theRatio.gain6Over1();
    gainRatios[1] = theRatio.gain6Over1()  * theRatio.gain12Over6();

    return gainRatios;
  }


  double fullScaleEnergy( const DetId & detId ) const 
  {
     return detId.subdetId() == EcalBarrel ? m_maxEneEB : m_maxEneEE ;
  }


  double peToAConversion( const DetId & detId ) const 
  {
     return detId.subdetId() == EcalBarrel ? m_peToABarrel : m_peToAEndcap ;
  }

    
  /// these fields are set in initializeEvent()
  const edm::Event * theEvent;
  const edm::EventPrincipal * theEventPrincipal;
  edm::ESHandle<EcalGainRatios> grHandle; 
  edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
 /// these come from the ParameterSet
  edm::InputTag theInputTag;
  edm::EDGetTokenT<COLLECTION> tok_;

  edm::ESHandle<ESGain>                hesgain      ;
  edm::ESHandle<ESMIPToGeVConstant>    hesMIPToGeV  ;
  edm::ESHandle<ESIntercalibConstants> hesMIPs      ;

  const ESGain*                esgain;
  const ESIntercalibConstants* esmips;
  const ESMIPToGeVConstant*    esMipToGeV;
  int ESgain ;
  double ESMIPToGeV; 

  double m_EBs25notCont;
  double m_EEs25notCont;

  double m_peToABarrel;
  double m_peToAEndcap;

  double m_maxEneEB ; // max attainable energy in the ecal barrel
  double m_maxEneEE ; // max attainable energy in the ecal endcap

  const EcalADCToGeVConstant* agc;
  const EcalIntercalibConstantsMC* ical;

  double theDefaultGains[NGAINS];

};

typedef EcalSignalGenerator<EBDigitizerTraits>   EBSignalGenerator;
typedef EcalSignalGenerator<EEDigitizerTraits>   EESignalGenerator;
typedef EcalSignalGenerator<ESDigitizerTraits>   ESSignalGenerator;

#endif

