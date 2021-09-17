#ifndef SimG4Core_CMSG4RegionReporter_H
#define SimG4Core_CMSG4RegionReporter_H

#include <string>

class CMSG4RegionReporter {
public:
  CMSG4RegionReporter();

  ~CMSG4RegionReporter();

  void ReportRegions(const std::string& ss);
};

#endif
