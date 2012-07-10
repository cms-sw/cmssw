#ifndef SimG4CMS_HcalTestHistoManager_H
#define SimG4CMS_HcalTestHistoManager_H
///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.h
// Histogram managing class for analysis in HcalTest
///////////////////////////////////////////////////////////////////////////////

//#include "DataSvc/Ref.h"
//#include "FileCatalog/IFileCatalog.h"
#include <memory>

//namespace pool {
//  class IDataSvc;
//}

#include <string>
class HcalTestHistoClass;

class HcalTestHistoManager {

public: 

  HcalTestHistoManager(const std::string &);
  virtual ~HcalTestHistoManager();

  void fillTree(HcalTestHistoClass *  histos);

private:
  //  pool::IFileCatalog            lcat;
  //std::auto_ptr<pool::IDataSvc> svc;
  
  //pool::Placement               placeH;
  //pool::Placement               placeVx;
  //pool::Ref<HcalTestHistoClass> h;
 
};

#endif
