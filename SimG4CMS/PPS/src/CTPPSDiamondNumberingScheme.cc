// //////////////////////
// Author
// Seyed Mohsen Etesami setesami@cern.ch
// ////////////////////////////

#include "SimG4CMS/PPS/interface/CTPPSDiamondNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

CTPPSDiamondNumberingScheme::CTPPSDiamondNumberingScheme()
{
  edm::LogInfo("CTPPSSimDiamond") << " Creating CTPPSDiamondNumberingScheme" << std::endl;
}

CTPPSDiamondNumberingScheme::~CTPPSDiamondNumberingScheme()
{
  edm::LogInfo("CTPPSSimDiamond") << " Deleting CTPPSDiamondNumberingScheme" << std::endl;
}
