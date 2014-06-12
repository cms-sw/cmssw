// File: MixingWorkerBase.cc
// Description:  see MixingWorkerBase.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "MixingWorkerBase.h"

namespace edm
{

  // Virtual destructor needed so MixingModule can delete the list
  // of MixingWorkers without knowing their exact type.
  MixingWorkerBase::~MixingWorkerBase() { }  

}//edm
