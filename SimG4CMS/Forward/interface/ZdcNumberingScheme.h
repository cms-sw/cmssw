#ifndef SimG4CMSForwardZdcNumberingScheme_h
#define SimG4CMSForwardZdcNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: ZdcNumberingScheme.h
// Date: 03.06
// Description: Numbering scheme for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include "G4Step.hh"

class ZdcNumberingScheme {
public:
  ZdcNumberingScheme(int);
  ~ZdcNumberingScheme() = default;

  void setVerbosity(const int);

  unsigned int getUnitID(const G4Step* aStep);

  /** pack the Unit ID for Zdc <br>
   *  z = 1,2 = -z,+z; subDet = 1,2,3 = EM,Lum,HAD; fiber = 1-96 (EM,HAD), 1 (Lum);
   *  channel = 1-5 (EM), layer# (Lum), 1-3 (HAD)
   */
  static unsigned int packZdcIndex(int subDet, int layer, int fiber, int channel, int z);

  // unpacking Unit ID for Zdc (-z=1, +z=2)
  static void unpackZdcIndex(const unsigned int& idx, int& subDet, int& layer, int& fiber, int& channel, int& z);

  int detectorLevel(const G4Step*);
  void detectorLevel(const G4Step*, int&, std::vector<int>&, std::vector<G4String>&);

private:
  int verbosity;
};

#endif
