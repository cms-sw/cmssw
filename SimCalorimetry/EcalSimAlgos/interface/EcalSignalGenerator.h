#ifndef EcalSimAlgos_EcalSignalGenerator_h
#define EcalSimAlgos_EcalSignalGenerator_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalBaseSignalGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "CalibFormats/EcalObjects/interface/EcalCoderDb.h"
#include "CalibFormats/EcalObjects/interface/EcalCalibrations.h"
#include "CalibFormats/EcalObjects/interface/EcalDbService.h"
#include "CalibFormats/EcalObjects/interface/EcalDbRecord.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
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

  EcalSignalGenerator(const edm::InputTag & inputTag, const edm::EDGetTokenT<COLLECTION> &t)
  : EcalBaseSignalGenerator(), theEvent(0), theEventPrincipal(0), theInputTag(inputTag), tok_(t) 
  { }

  virtual ~EcalSignalGenerator() {}


  void initializeEvent(const edm::Event * event, const edm::EventSetup * eventSetup)
  {
    theEvent = event;
    eventSetup->get<EcalDbRecord>().get(theConditions);
    theParameterMap->setDbService(theConditions.product());
  }

  /// some users use EventPrincipals, not Events.  We support both
  void initializeEvent(const edm::EventPrincipal * eventPrincipal, const edm::EventSetup * eventSetup)
  {
    theEventPrincipal = eventPrincipal;
    eventSetup->get<EcalDbRecord>().get(theConditions);
    theParameterMap->setDbService(theConditions.product());
  }

  virtual void fill(edm::ModuleCallingContext const* mcc)
  {

    std::cout << " In Signal Generator, Filling event " << std::endl;

    theNoiseSignals.clear();
    edm::Handle<COLLECTION> pDigis;
    const COLLECTION *  digis = 0;
    // try accessing by whatever is set, Event or EventPrincipal
    if(theEvent) 
     {
      if( theEvent->getByToken(tok_, pDigis) ) {
        digis = pDigis.product(); // get a ptr to the product
        LogTrace("EcalSignalGenerator") << "total # digis  for "  << theInputTag << " " <<  digis->size();
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
        // for the first signal, set the starting cap id
        if((it == digis->begin()) && theElectronicsSim)
        {
          int startingCapId = (*it)[0].capid();
          theElectronicsSim->setStartingCapId(startingCapId);
          theParameterMap->setFrameSize(it->id(), it->size());
        }
	// need to convert to something useful
        theNoiseSignals.push_back(samplesInPE(*it));
      }
    }
  }

private:

  CaloSamples samplesInPE(const DIGI & digi)
  {
    // calibration, for future reference:  (same block for all Ecal types)
    //EcalDetId cell = digi.id();
    //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
    //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
    //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
    //EcalCoderDb coder (*channelCoder, *channelShape);
    //CaloSamples result;
    //coder.adc2fC(digi, result);
    //fC2pe(result);

    std::cout << " EcalSignalGenerator: noise result " << result << std::endl;



    return result;
  }

    
  /// these fields are set in initializeEvent()
  const edm::Event * theEvent;
  const edm::EventPrincipal * theEventPrincipal;
  edm::ESHandle<EcalDbService> theConditions;
  /// these come from the ParameterSet
  edm::InputTag theInputTag;
  edm::EDGetTokenT<COLLECTION> tok_;
};

typedef EcalSignalGenerator<EBDigitizerTraits>   EBSignalGenerator;
typedef EcalSignalGenerator<EEDigitizerTraits>   EESignalGenerator;
typedef EcalSignalGenerator<ESDigitizerTraits>   ESSignalGenerator;

#endif

