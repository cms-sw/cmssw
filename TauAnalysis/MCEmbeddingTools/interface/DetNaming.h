#ifndef TauAnalysis_MCEmbeddingTools_DetNaming_h
#define TauAnalysis_MCEmbeddingTools_DetNaming_h

#include <vector>
#include <string>
#include <map>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <boost/foreach.hpp>

class DetNaming 
{
 public:
  DetNaming();
  ~DetNaming() {}

  std::string getKey(const DetId&);
  std::vector<std::string> getAllKeys();

 private:
  typedef std::map<int, std::string > TMyMainMap;
  typedef std::map<int, std::map<int, std::string> > TMySubMap;
  TMyMainMap detMap_;
  TMySubMap  subDetMap_;
};


#endif
