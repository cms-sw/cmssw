#ifndef MU_END_WIRE_ELECTRONICS_SIM_H
#define MU_END_WIRE_ELECTRONICS_SIM_H

/** \class CSCWireElectronicsSim
 * Model the readout electronics chain for EMU CSC wires
 *
 * \author Rick Wilkinson
 *
 */

#include "SimMuon/CSCDigitizer/src/CSCBaseElectronicsSim.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

// declarations
class CSCLayer;
class CSCDetectorHit;
class CSCWireDigi;
class CSCAnalogSignal;


class CSCWireElectronicsSim : public CSCBaseElectronicsSim
{
public:
  /// configurable parameters
  CSCWireElectronicsSim(const edm::ParameterSet &p);

  void setFraction(float newFraction)  {theFraction = newFraction;};

  void fillDigis(CSCWireDigiCollection & digis);

private:
  /// initialization for each layer
  virtual void initParameters();

  // will return wire group, given wire.
  virtual int readoutElement(int element) const;

  float calculateAmpResponse(float t) const;
 
  virtual float timeOfFlightCalibration(int wireGroup) const;

  /// we code strip indices from 1-80, and wire indices start at 100
  virtual int channelIndex(int channel) const {return channel+100;}

  // member data
  // the fractional discriminator returns the time when the signal
  // reaches this fraction of its maximum
  float theFraction;
  float theWireNoise;
  float theWireThreshold;

};

#endif
