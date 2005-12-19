///////////////////////////////////////////////////////////////////////////////
// File: ShowerForwardNumberingScheme.h
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////
#ifndef ShowerForwardNumberingScheme_h
#define ShowerForwardNumberingScheme_h

#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

class ShowerForwardNumberingScheme : public EcalNumberingScheme {

public:

  ShowerForwardNumberingScheme(int);
  ~ShowerForwardNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const;
  void findXY(const int& layer, const int& waf, int& x, int& y) const;

private:

  int iquad_max[40];
  int iquad_min[40];
  int Ntot[40];
  int Ncols[40];

};

#endif
