#ifndef SimG4Core_G4RegionReporter_H
#define SimG4Core_G4RegionReporter_H

#include <string>

class G4RegionReporter
{
public:

  G4RegionReporter();
  
  ~G4RegionReporter();

  void ReportRegions(const std::string& ss);

};

#endif
