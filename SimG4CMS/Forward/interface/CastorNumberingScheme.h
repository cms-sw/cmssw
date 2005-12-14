///////////////////////////////////////////////////////////////////////////////
// File: CastorNumberingScheme.h
// Date: 02.04
// Description: Numbering scheme for Castor
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef CastorNumberingScheme_h
#define CastorNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"
#include <boost/cstdint.hpp>

class CastorNumberingScheme : public CaloNumberingScheme {

public:

  CastorNumberingScheme(int);
  ~CastorNumberingScheme();

  virtual uint32_t getUnitID(const G4Step* aStep) const;

  /** pack the Unit ID for  Castor <br>
   Bits  0- 5: zmodule index <br>
   Bits  6- 9: sector  index <br>
   Bits 10-19: unused        <br>
   Bit     20: +/- z side    <br>
   Bits 22-27: unused        <br>
   Bits 28-31: subdetector   <br>
   *  (+z=1,-z=2);  sector=1..16, zmodule=1..18;
   */
  static uint32_t packIndex(int det, int z, int sector, int zmodule);
  static void   unpackIndex(const uint32_t& idx, int& det, int& z, 
			    int& sector, int& zmodule);

private:

  int zsideScale;
  int sectorScale;

};

#endif
