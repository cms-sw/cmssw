#ifndef NeutronReader_h
#define NeutronReader_h
/**
 * interface to methods which return a set of SimHits in a chamber
 \author Rick Wilkinson
 */

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include<vector>

class NeutronReader
{
public:  
  NeutronReader() {};
  virtual ~NeutronReader() {};

  virtual void readNextEvent(int chamberType, edm::PSimHitContainer & result) = 0;
};

#endif

