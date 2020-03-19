// //////////////////////
// Author
// Seyed Mohsen Etesami setesami@cern.ch
// ////////////////////////////

#include "SimG4CMS/PPS/interface/PPSDiamondNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

PPSDiamondNumberingScheme::PPSDiamondNumberingScheme() {
  edm::LogInfo("PPSSimDiamond") << " Creating PPSDiamondNumberingScheme" << std::endl;
}

PPSDiamondNumberingScheme::~PPSDiamondNumberingScheme() {
  edm::LogInfo("PPSSimDiamond") << " Deleting PPSDiamondNumberingScheme" << std::endl;
}
