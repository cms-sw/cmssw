#include "SimG4CMS/PPS/interface/PPSStripNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

PPSStripNumberingScheme::PPSStripNumberingScheme(int i) {
  edm::LogInfo("TotemRP") << " Creating PPSStripNumberingScheme" << std::endl;
}

PPSStripNumberingScheme::~PPSStripNumberingScheme() {
  edm::LogInfo("TotemRP") << " Deleting PPSStripNumberingScheme" << std::endl;
}
