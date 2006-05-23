#ifndef HcalTestBeam_HcalTB02HistoManager_H
#define HcalTestBeam_HcalTB02HistoManager_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HistoManager
//
/**\class HcalTB02HistoManager HcalTB02HistoManager.h SimG4CMS/HcalTestBeam/interface/HcalTB02HistoManager.h
  
 Description: Manages Root file creation for Hcal Test Beam 2002 studies
  
 Usage: Used in 2002 Hcal Test Beam studies
*/
//
// Original Author:  
//         Created:  Thu Sun 21 10:14:34 CEST 2006
// $Id$
//
  
// system include files
#include <string>

// user include files
#include "DataSvc/Ref.h"
#include "FileCatalog/IFileCatalog.h"

namespace pool {
  class IDataSvc;
}

class HcalTB02HistoClass;

class HcalTB02HistoManager {

public: 

  HcalTB02HistoManager(const std::string &);
  virtual ~HcalTB02HistoManager();

  void fillTree(HcalTB02HistoClass *  histos);

private:
  pool::IFileCatalog             lcat;
  std::auto_ptr<pool::IDataSvc>  svc;
  
  pool::Placement                placeH;
  pool::Placement                placeVx;
  pool::Ref<HcalTB02HistoClass> h;
 
};

#endif
