// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT1NumberingScheme
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemT1NumberingScheme.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemT1NumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
TotemT1NumberingScheme::TotemT1NumberingScheme(int i) {

  edm::LogInfo("ForwardSim") << " Creating TotemT1NumberingScheme";
  SetCurrentDetectorPosition(i);
}

TotemT1NumberingScheme::~TotemT1NumberingScheme() {
  edm::LogInfo("ForwardSim") << " Deleting TotemT1NumberingScheme";
}
