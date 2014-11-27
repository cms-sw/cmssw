#ifndef _hgcfeelectronics_h_
#define _hgcfeelectronics_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class HGCFEElectronics
   @short models the behavior of the front-end electronics
 */

template <class D>
class HGCFEElectronics
{
 public:

  enum HGCFEElectronicsFirmwareVersion { SIMPLE, WITHTOT };
  
  /**
     @short CTOR
   */
  HGCFEElectronics(const edm::ParameterSet &ps,float bxTime);

  /**
     @short switches according to the firmware version
   */
  inline void runShaper(D &dataFrame)
  {
    switch(fwVersion_)
      {
      case WITHTOT : { runShaperWithToT(dataFrame); break; }
      default      : { runSimpleShaper(dataFrame);  break; }
      }
  }

  /**
     @short returns the LSB in MIP currently configured
   */
  float getLSB() { return lsbInMIP_; }  

  /**
     @short DTOR
   */
  ~HGCFEElectronics() {}

private:

  /**
     @short applies a CR-RC like-shape to each time sample and propagates the tails to the subsequent time samples
   */
  void runSimpleShaper(D &dataFrame);

  /**
     @short implements pulse shape and switch to time over threshold including deadtime
   */
  void runShaperWithToT(D &dataFrame);

  //private members
  uint32_t fwVersion_;
  float shaperN_, shaperTau_;
  float lsbInMIP_, mipInfC_, gainChangeInfC_;
  float bxTime_;
};

//#include "SimCalorimetry/HGCSimProducers/src/HGCFEElectronics.cc"

#endif
