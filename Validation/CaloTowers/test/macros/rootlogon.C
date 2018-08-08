#include "TSystem.h"

#include "rootlogon.h"

void rootlogon()
{
    setColors();
    
    gSystem->Load("libFWCoreFWLite.so");
    AutoLibraryLoader::enable();
}
