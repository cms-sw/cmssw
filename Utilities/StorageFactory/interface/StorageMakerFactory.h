#ifndef Utilities_StorageFactory_StorageMakerFactory_h
#define Utilities_StorageFactory_StorageMakerFactory_h
// -*- C++ -*-
//
// Package:     StorageFactory
// Class  :     StorageMakerFactory
// 
/**\class StorageMakerFactory StorageMakerFactory.h Utilities/StorageFactory/interface/StorageMakerFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Fri Apr 13 18:07:17 EDT 2007
// $Id$
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"


// forward declarations
class StorageMaker;
namespace edm {
   namespace storage {
      typedef edmplugin::PluginFactory<StorageMaker *(void)> StorageMakerFactory;
   }
}
#endif
