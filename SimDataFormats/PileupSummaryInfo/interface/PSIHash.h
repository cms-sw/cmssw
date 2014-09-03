#ifndef PSIHash_h
#define PSIHash_h

#include "PileupSummaryInfo.h"
#include <vector>
#include <string>

namespace sim{
size_t PSIHash(const std::vector<PileupSummaryInfo> &psi ) {
  
  std::hash<std::string> str_hash;

  std::stringstream result;
  for ( unsigned int i=0; i<psi.size(); i++)
    result << psi[i].getPU_NumInteractions() << " ";
  
  return str_hash(result.str());
}
}
#endif
