#ifndef EcalZeroSuppressionAlgos_EcalZeroSuppressor_h
#define EcalZeroSuppressionAlgos_EcalZeroSuppressor_h

/*
 * \file EcalZeroSuppressor.h
 *
 * $Date: 2011/05/20 17:17:34 $
 * $Revision: 1.5 $
 * \author F. Cossutti
 *
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "SimCalorimetry/EcalZeroSuppressionAlgos/interface/TrivialAmplitudeAlgo.h"

#include <vector>

template<class C> class EcalZeroSuppressor 
{
public:
  
  virtual ~EcalZeroSuppressor(){};

  // the threshold has to be expressed in number of noise sigmas for ADC counts in the highest gain

  // bool accept(const C& frame, const double & threshold);
  bool accept(const C& frame, const double & threshold);
  
  /// can be fetched every event from the EventSetup
  void setPedestals(const EcalPedestals * pedestals) {thePedestals = pedestals;} 

 private:

  const EcalPedestals * thePedestals;
 
  void findGain12Pedestal(const DetId & detId, 
                          double & pedestal, double & width);
  
  TrivialAmplitudeAlgo<C> theEnergy_;
  
};

#include "EcalZeroSuppressor.icc"
#endif
