#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/HepPDTESSource/interface/HepPDTESSource.h"
#include "SimGeneral/HepPDTESSource/interface/PythiaPDTESSource.h"


  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE( HepPDTESSource );
  DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE( PythiaPDTESSource );
