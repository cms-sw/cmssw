///////////////////////////////////////////////////////////////////////////////
// File: HcalTBNumberingScheme.h
// Description: Numbering scheme for hadron calorimeter in test beam
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTBNumberingPacker_h
#define HcalTBNumberingPacker_h

#include <boost/cstdint.hpp>
#include <vector>

class HcalTBNumberingScheme {

public:
  HcalTBNumberingScheme() {}
  virtual ~HcalTBNumberingScheme() {}
	 
  static uint32_t              getUnitID (const uint32_t id, const int mode);
  static std::vector<uint32_t> getUnitIDs(const int type, const int mode);
};

#endif
