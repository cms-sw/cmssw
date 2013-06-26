#ifndef SimDataFormats_HcalTestNumbering_h
#define SimDataFormats_HcalTestNumbering_h
///////////////////////////////////////////////////////////////////////////////
// File: HcalTestNumbering.h
// Description: Numbering scheme for hadron calorimeter (detailed for TB)
///////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>

class HcalTestNumbering {

public:
  HcalTestNumbering() {}
  virtual ~HcalTestNumbering() {}
  static uint32_t  packHcalIndex(int det, int z, int depth, int eta, 
				 int phi, int lay);
  static void      unpackHcalIndex(const uint32_t & idx, int& det, int& z, 
				   int& depth, int& eta, int& phi, int& lay);
};

#endif
