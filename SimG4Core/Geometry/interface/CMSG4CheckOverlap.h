#ifndef SimG4Core_CMSG4CheckOverlap_H
#define SimG4Core_CMSG4CheckOverlap_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>

class CustomUIsession;
class G4VPhysicalVolume;

class CMSG4CheckOverlap {
public:
  CMSG4CheckOverlap(edm::ParameterSet const& p, std::string& regFile, CustomUIsession*, G4VPhysicalVolume* world);
  ~CMSG4CheckOverlap() = default;

private:
  void makeReportForMaterials(std::ofstream& fout);
  void makeReportForGeometry(std::ofstream& fout, G4VPhysicalVolume* world);
  void makeReportForOverlaps(std::ofstream& fout, const edm::ParameterSet& p, G4VPhysicalVolume* world);
};

#endif
