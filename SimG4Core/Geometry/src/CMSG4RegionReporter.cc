#include "SimG4Core/Geometry/interface/CMSG4RegionReporter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4LogicalVolume.hh"
#include "G4ProductionCuts.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <iomanip>
#include <fstream>

CMSG4RegionReporter::CMSG4RegionReporter() {}

CMSG4RegionReporter::~CMSG4RegionReporter() {}

void CMSG4RegionReporter::ReportRegions(const std::string& ss) {
  std::ofstream fout(ss.c_str(), std::ios::out);
  if (fout.fail()) {
    edm::LogWarning("SimG4CoreGeometry") << "CMSG4RegionReporter: file <" << ss
                                         << "> is not opened - no report provided";
    return;
  }
  G4RegionStore* regStore = G4RegionStore::GetInstance();

  unsigned int numRegions = regStore->size();

  unsigned int i;

  fout << "\n";
  fout << "#---------------------------------------------------------------------";
  fout << "------------------------------------"
       << "\n";
  fout << "## List of Regions, root logical volumes and cuts. "
       << "\n";
  fout << "## Number of regions = " << numRegions << "\n";

  //  Banner
  fout << "# " << std::setw(24) << " Region, " << std::setw(38) << " LogicalVolume, "
       << " Cuts:Gamma, Electron, Positron, Proton, Units"
       << "\n";
  fout << "#---------------------------------------------------------------------";
  fout << "------------------------------------"
       << "\n";

  for (i = 0; i < numRegions; ++i) {
    G4Region* region = regStore->at(i);
    G4ProductionCuts* prodCuts = region->GetProductionCuts();

    G4LogicalVolume* lv;

    G4double lengthUnit = CLHEP::mm;
    G4String lengthUnitName = "mm";
    unsigned int pmax = 4;  // g, e-, e+, proton

    std::vector<G4LogicalVolume*>::iterator rootLVItr = region->GetRootLogicalVolumeIterator();
    size_t numRootLV = region->GetNumberOfRootVolumes();

    for (size_t iLV = 0; iLV < numRootLV; ++iLV, ++rootLVItr) {
      // Cover each root logical volume in this region

      //Set the couple to the proper logical volumes in that region
      lv = *rootLVItr;

      // fout << " Region=" << region->GetName()
      //     << " Logical-Volume = " << lv->GetName();
      char quote = '"';
      std::ostringstream regName, lvName;
      regName << quote << region->GetName() << quote;
      lvName << quote << lv->GetName() << quote;
      fout << " " << std::setw(26) << regName.str() << " ,";
      fout << " " << std::setw(36) << lvName.str() << " ,";

      unsigned int ic;
      for (ic = 0; ic < pmax; ++ic) {
        G4double cutLength = prodCuts->GetProductionCut(ic);
        fout << " " << std::setw(5) << cutLength / lengthUnit;
        if (ic < pmax - 1) {
          fout << " , ";
        } else {
          fout << " ,   " << lengthUnitName;
        }
      }
      fout << "\n";
    }
  }
  fout << "#---------------------------------------------------------------------";
  fout << "------------------------------------"
       << "\n";
  fout << "\n";
  fout.close();
}
