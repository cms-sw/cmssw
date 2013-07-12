#ifndef EcalZeroSuppressionAlgos_TrivialAmplitudeAlgo_h
#define EcalZeroSuppressionAlgos_TrivialAmplitudeAlgo_h

/*
 * \file TrivialAmplitudeAlgo.h
 *
 * $Date: 2007/03/07 10:12:43 $
 * $Revision: 1.5 $
 * \author F. Cossutti
 *
 * simple weights algorithm for energy and pedestal reconstruction
 *
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <vector>

template<class C> class TrivialAmplitudeAlgo 
{
public:

  explicit TrivialAmplitudeAlgo();
  virtual ~TrivialAmplitudeAlgo(){};

  double energy(const C& frame);

  double pedestal(const C& frame);

 private:

  std::vector<double> theWeights;
  std::vector<double> theGainFactors;

};

#include "TrivialAmplitudeAlgo.icc"
#endif
