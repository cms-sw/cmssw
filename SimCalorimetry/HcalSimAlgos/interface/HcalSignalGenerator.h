#ifndef HcalSimAlgos_HcalSignalGenerator_h
#define HcalSimAlgos_HcalSignalGenerator_h

#include "SimCalorimetry/HcalSimAlgos/interface/HcalBaseSignalGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** Converts digis back into analog signals, to be used
 *  as noise 
 */

#include <iostream>

namespace edm {
  class ModuleCallingContext;
}

template<class HCALDIGITIZERTRAITS>
class HcalSignalGenerator : public HcalBaseSignalGenerator
{
public:
  typedef typename HCALDIGITIZERTRAITS::Digi DIGI;
  typedef typename HCALDIGITIZERTRAITS::DigiCollection COLLECTION;

  HcalSignalGenerator():HcalBaseSignalGenerator() { }

  HcalSignalGenerator(const edm::InputTag & inputTag, const edm::EDGetTokenT<COLLECTION> &t)
  : HcalBaseSignalGenerator(), theEvent(0), theEventPrincipal(0), theInputTag(inputTag), tok_(t) 
  { }

  virtual ~HcalSignalGenerator() {}


  void initializeEvent(const edm::Event * event, const edm::EventSetup * eventSetup)
  {
    theEvent = event;
    eventSetup->get<HcalDbRecord>().get(theConditions);
    theParameterMap->setDbService(theConditions.product());
  }

  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal * eventPrincipal, const edm::EventSetup * eventSetup)
  {
    theEventPrincipal = eventPrincipal;
    eventSetup->get<HcalDbRecord>().get(theConditions);
    theParameterMap->setDbService(theConditions.product());
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
        LogTrace("HcalSignalGenerator") << "total # digis  for "  << theInputTag << " " <<  digis->size();
      }
      else
      {
        throw cms::Exception("HcalSignalGenerator") << "Cannot find input data " << theInputTag;
      }
    }
    else if(theEventPrincipal)
    {
       std::shared_ptr<edm::Wrapper<COLLECTION>  const> digisPTR =
	 edm::getProductByTag<COLLECTION>(*theEventPrincipal, theInputTag, mcc );
       if(digisPTR) {
          digis = digisPTR->product();
       }
    }
    else
    {
      throw cms::Exception("HcalSignalGenerator") << "No Event or EventPrincipal was set";
    }

    if (digis)
    {

      // loop over digis, adding these to the existing maps
      for(typename COLLECTION::const_iterator it  = digis->begin();
          it != digis->end(); ++it) 
      {
        // for the first signal, set the starting cap id
        if((it == digis->begin()) && theElectronicsSim)
        {
          int startingCapId = (*it)[0].capid();
          theElectronicsSim->setStartingCapId(startingCapId);
          // theParameterMap->setFrameSize(it->id(), it->size()); //don't need this
        }
	if(validDigi(*it)) {
	  theNoiseSignals.push_back(samplesInPE(*it));
	}
      }
    }
  }

private:


  virtual void fillNoiseSignals(CLHEP::HepRandomEngine*) override {}
  virtual void fillNoiseSignals() override {}

  bool validDigi(const DIGI & digi)
  {
    int DigiSum = 0;
    for(int id = 0; id<digi.size(); id++) {
      if(digi[id].adc() > 0) ++DigiSum;
    }
    return(DigiSum>0);
  }


  CaloSamples samplesInPE(const DIGI & digi)
  {

    // For PreMixing, (Note that modifications will need to be made for DataMixing) the 
    // energy for each channel is kept as fC*10, but stored as an integer in ADC.  If this
    // results in an overflow, the "standard" ADC conversion is used and that channel is marked
    // with an error that allows the "standard" decoding to convert ADC back to fC.  So, most
    // channels get to fC by just dividing ADC/10; some require special treatment.

    // calibration, for future reference:  (same block for all Hcal types)
    HcalDetId cell = digi.id();
    //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
    //const HcalQIECoder* channelCoder = theConditions->getHcalCoder (cell);
    //const HcalQIEShape* channelShape = theConditions->getHcalShape (cell);
    //HcalCoderDb coder (*channelCoder, *channelShape);
    CaloSamples result = CaloSamples(cell,digi.size());;
    //coder.adc2fC(digi, result);

    // first, check if there was an overflow in this fake digi:
    bool overflow = false;
    // find and list them

    for(int isample=0; isample<digi.size(); ++isample) {
      if(digi[isample].er()) overflow = true; 
    }
 
    if(overflow) {  // do full conversion, go back and overwrite fake entries
 
      const HcalQIECoder* channelCoder = theConditions->getHcalCoder (cell);
      const HcalQIEShape* channelShape = theConditions->getHcalShape (cell);
      HcalCoderDb coder (*channelCoder, *channelShape);
      coder.adc2fC(digi, result);
 
      // overwrite with coded information
      for(int isample=0; isample<digi.size(); ++isample) {
	if(!digi[isample].er()) result[isample] = float(digi[isample].adc())/10.;
      }
    }
    else {  // saves creating the coder, etc., every time
      // use coded information
      for(int isample=0; isample<digi.size(); ++isample) {
	result[isample] = float(digi[isample].adc())/10.;
      }
      result.setPresamples(digi.presamples());
    }
 
    // std::cout << " HcalSignalGenerator: noise input ADC " << digi << std::endl;
    // std::cout << " HcalSignalGenerator: noise input in fC " << result << std::endl;
 
    // translation done in fC, convert to pe: 
    fC2pe(result);

    return result;
  }
    
  /// these fields are set in initializeEvent()
  const edm::Event * theEvent;
  const edm::EventPrincipal * theEventPrincipal;
  edm::ESHandle<HcalDbService> theConditions;
  /// these come from the ParameterSet
  edm::InputTag theInputTag;
  edm::EDGetTokenT<COLLECTION> tok_;
};

typedef HcalSignalGenerator<HBHEDigitizerTraits> HBHESignalGenerator;
typedef HcalSignalGenerator<HODigitizerTraits>   HOSignalGenerator;
typedef HcalSignalGenerator<HFDigitizerTraits>   HFSignalGenerator;
typedef HcalSignalGenerator<ZDCDigitizerTraits>  ZDCSignalGenerator;

#endif

