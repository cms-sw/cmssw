#include "SimMuon/Neutron/src/RootNeutronReader.h"
#include <sstream>

RootNeutronReader::RootNeutronReader(const std::string& fileName) : theFile(new TFile(fileName.c_str())) {}

RootChamberReader& RootNeutronReader::chamberReader(int chamberType) {
  std::map<int, RootChamberReader>::iterator mapItr = theChamberReaders.find(chamberType);

  if (mapItr != theChamberReaders.end()) {
    return mapItr->second;
  } else {
    // make a new one
    std::ostringstream treeName;
    treeName << "ChamberType" << chamberType;
    theChamberReaders[chamberType] = RootChamberReader(theFile, treeName.str());
    return theChamberReaders[chamberType];
  }
}

void RootNeutronReader::readNextEvent(int chamberType, edm::PSimHitContainer& result) {
  chamberReader(chamberType).read(result);
}
