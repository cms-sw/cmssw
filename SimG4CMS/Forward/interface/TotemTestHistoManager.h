#ifndef Forward_TotemTestHistoManager_h
#define Forward_TotemTestHistoManager_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestHistoManager
//
/**\class TotemTestHistoManager TotemTestHistoManager.h SimG4CMS/Forward/interface/TotemTestHistoManager.h
 
 Description: Manages Root file creation for Totem Tests
 
 Usage:
    Used in testing Totem simulation
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id$
//
 
// system include files

// user include files
#include "DataSvc/Ref.h"
#include "FileCatalog/IFileCatalog.h"
#include <string>

namespace pool {
  class IDataSvc;
}

class TotemTestHistoClass;

class TotemTestHistoManager {

public: 

  TotemTestHistoManager(const std::string &);
  virtual ~TotemTestHistoManager();

  void fillTree(TotemTestHistoClass *  histos);

private:
  pool::IFileCatalog             lcat;
  std::auto_ptr<pool::IDataSvc>  svc;
  
  pool::Placement                placeH;
  pool::Placement                placeVx;
  pool::Ref<TotemTestHistoClass> h;
 
};

#endif
