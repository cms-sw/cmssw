#include "TSystem.h"
#include "rootlogon.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

void rootlogon()
{
    setColors();
    gSystem->Load("libFWCoreFWLite.so");
    FWLiteEnabler::enable();    
}
