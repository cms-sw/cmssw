#ifndef CastorSim_CastorElectronicsSim_h
#define CastorSim_CastorElectronicsSim_h
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Random/RandFlat.h"

class CastorDataFrame;

class CastorAmplifier;
class CastorCoderFactory;

class CastorElectronicsSim {
public:
  CastorElectronicsSim(CastorAmplifier * amplifier, 
                     const CastorCoderFactory * coderFactory);
  ~CastorElectronicsSim();

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  void analogToDigital(CaloSamples & linearFrame, CastorDataFrame & result);

  /// Things that need to be initialized every event
void newEvent();

private:
  template<class Digi> void convert(CaloSamples & frame, Digi & result);

  CastorAmplifier * theAmplifier;
  const CastorCoderFactory * theCoderFactory;
  CLHEP::RandFlat * theRandFlat;

  int theStartingCapId;
};

  
#endif
  
