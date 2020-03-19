#ifndef CastorSim_CastorElectronicsSim_h
#define CastorSim_CastorElectronicsSim_h

/** This class turns a CaloSamples, representing the analog
    signal input to the readout electronics, into a
    digitized data frame
 */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class CastorDataFrame;

class CastorAmplifier;
class CastorCoderFactory;

namespace CLHEP {
  class HepRandomEngine;
}

class CastorElectronicsSim {
public:
  CastorElectronicsSim(CastorAmplifier *amplifier, const CastorCoderFactory *coderFactory);
  ~CastorElectronicsSim();

  void analogToDigital(CLHEP::HepRandomEngine *, CaloSamples &linearFrame, CastorDataFrame &result);

  /// Things that need to be initialized every event
  void newEvent(CLHEP::HepRandomEngine *);

private:
  template <class Digi>
  void convert(CaloSamples &frame, Digi &result, CLHEP::HepRandomEngine *);

  CastorAmplifier *theAmplifier;
  const CastorCoderFactory *theCoderFactory;

  int theStartingCapId;
};

#endif
