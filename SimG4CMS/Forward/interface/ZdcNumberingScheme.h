///////////////////////////////////////////////////////////////////////////////
// File: ZdcNumberingScheme.h
// Date: 03.06
// Description: Numbering scheme for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#undef debug
#ifndef ZdcNumberingScheme_h
#define ZdcNumberingScheme_h

#include "G4Step.hh"

class ZdcNumberingScheme {
public:
  ZdcNumberingScheme(int);
  virtual ~ZdcNumberingScheme();

  void setVerbosity(const int);

  virtual unsigned int getUnitID(const G4Step* aStep) const;

  /** pack the Unit ID for Zdc <br>
   *  z = 1,2 = -z,+z; subDet = 1,2,3 = EM,Lum,HAD; fiber = 1-96 (EM,HAD), 1 (Lum);
   *  channel = 1-5 (EM), layer# (Lum), 1-3 (HAD)
   */
  static unsigned int packZdcIndex(int subDet, int layer, int fiber, int channel, int z);

  // unpacking Unit ID for Zdc (-z=1, +z=2)
  static void unpackZdcIndex(const unsigned int& idx, int& subDet, int& layer, int& fiber, int& channel, int& z);

private:
  int verbosity;
};

#endif
