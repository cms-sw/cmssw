///////////////////////////////////////////////////////////////////////////////
// File: TBHcalNumberingScheme.h
// Description: Numbering scheme for hadron calorimeter in test beam
///////////////////////////////////////////////////////////////////////////////
#ifndef TBHcalNumberingPacker_h
#define TBHcalNumberingPacker_h

#include <boost/cstdint.hpp>
#include <vector>

class TBHcalNumberingScheme {

public:
  TBHcalNumberingScheme(int iv=0): verbosity(iv) {}
  virtual ~TBHcalNumberingScheme() {}
	 
  uint32_t              getUnitID (const uint32_t id, const int mode);
  std::vector<uint32_t> getUnitIDs(const int type, const int mode);

private:
  int verbosity;
};

#endif
