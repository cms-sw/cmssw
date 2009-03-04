#ifndef HcalSimAlgos_HcalSignalGenerator_h
#define HcalSimAlgos_HcalSignalGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/Common/interface/Handle.h"

/** Converts digis back into analog signals, to be used
 *  as noise 
 */

#include <iostream>

template<class HCALDIGITIZERTRAITS>
class HcalSignalGenerator : public CaloVNoiseSignalGenerator
{
public:
  typedef typename HCALDIGITIZERTRAITS::Digi DIGI;
  typedef typename HCALDIGITIZERTRAITS::DigiCollection COLLECTION;

  HcalSignalGenerator(const edm::InputTag & inputTag)
  : theInputTag(inputTag) {}

  virtual ~HcalSignalGenerator() {}

  void initializeEvent(const edm::Event * event, const edm::EventSetup * eventSetup)
  {
    theEvent = event;
    eventSetup->get<HcalDbRecord>().get(theConditions);
    theShape = theConditions->getHcalShape (); // this one is generic
  }

  void setParameterMap(HcalSimParameterMap * newMap) {theParameterMap = *newMap;}

protected:

  virtual void fillNoiseSignals()
  {
    theNoiseSignals.clear();
    edm::Handle<COLLECTION> pDigis;
    const COLLECTION *  digis = 0;
    if( theEvent->getByLabel(theInputTag, pDigis) ) {
      digis = pDigis.product(); // get a ptr to the product
      LogTrace("HcalSignalGenerator") << "total # digis  for "  << theInputTag << " " <<  digis->size();
    }
    else
    {
       throw cms::Exception("HcalSignalGenerator") << "Cannot find input data " << theInputTag;
    }

    if (digis)
    {
      // loop over digis, adding these to the existing maps
      for(typename COLLECTION::const_iterator it  = digis->begin();
          it != digis->end(); ++it) 
      {
        theNoiseSignals.push_back(samplesInPE(*it));
      }
    }
  }

private:

  CaloSamples samplesInPE(const DIGI & digi)
  {
    // calibration, for future reference:  (same block for all Hcal types)
    HcalDetId cell = digi.id();
    //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);
    const HcalQIECoder* channelCoder = theConditions->getHcalCoder (cell);
    HcalCoderDb coder (*channelCoder, *theShape);

    CaloSamples result;
    coder.adc2fC(digi, result);
    fC2pe(result);
    return result;
  }
 

  void fC2pe(CaloSamples & samples) const
  {
    float factor = 1./theParameterMap.simParameters(samples.id()).photoelectronsToAnalog();
    samples *= factor;
  }

  /// these fields are set in initializeEvent()
  const edm::Event * theEvent;
  edm::ESHandle<HcalDbService> theConditions;
  const HcalQIEShape* theShape;

  /// these come from the ParameterSet
  edm::InputTag theInputTag;
  /// use hardcoded defaults unless overridden
  HcalSimParameterMap theParameterMap;
};


typedef HcalSignalGenerator<HBHEDigitizerTraits> HBHESignalGenerator;
typedef HcalSignalGenerator<HODigitizerTraits>   HOSignalGenerator;
typedef HcalSignalGenerator<HFDigitizerTraits>   HFSignalGenerator;
typedef HcalSignalGenerator<ZDCDigitizerTraits>  ZDCSignalGenerator;



#endif



