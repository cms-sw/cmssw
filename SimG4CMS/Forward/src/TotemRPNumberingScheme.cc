// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemRPNumberingScheme
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemRPNumberingScheme.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemRPNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
TotemRPNumberingScheme::TotemRPNumberingScheme(int i) {
  edm::LogInfo("ForwardSim") << " Creating TotemRPNumberingScheme";
}

TotemRPNumberingScheme::~TotemRPNumberingScheme() {
  edm::LogInfo("ForwardSim") << " Deleting TotemRPNumberingScheme";
}
