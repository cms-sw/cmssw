///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.h
// Histogram managing class for analysis in HcalTest
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTestHistoManager_H
#define HcalTestHistoManager_H

#include "DataSvc/Ref.h"
#include "FileCatalog/IFileCatalog.h"

namespace pool {
  class IDataSvc;
}

#include <string>
class HcalTestHistoClass;

class HcalTestHistoManager {

public: 

  HcalTestHistoManager(int, const std::string &);
  virtual ~HcalTestHistoManager();

  void fillTree(HcalTestHistoClass *  histos);

private:
  int                           verbosity;
  pool::IFileCatalog            lcat;
  std::auto_ptr<pool::IDataSvc> svc;
  
  pool::Placement               placeH;
  pool::Placement               placeVx;
  pool::Ref<HcalTestHistoClass> h;
 
};

#endif
