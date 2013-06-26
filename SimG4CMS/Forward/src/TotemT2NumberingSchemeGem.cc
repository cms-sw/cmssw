// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT2NumberingSchemeGem
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemT2NumberingSchemeGem.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemT2NumberingSchemeGem.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
TotemT2NumberingSchemeGem::TotemT2NumberingSchemeGem(int i) {

  edm::LogInfo("ForwardSim") << " Creating TotemT2NumberingSchemeGem";
}

TotemT2NumberingSchemeGem::~TotemT2NumberingSchemeGem() {
  edm::LogInfo("ForwardSim") << " Deleting TotemT2NumberingSchemeGem";
}
