///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.cc
// Description: Histogram managing class in HcalTestAnalysis (HcalTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoManager.h"
#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
//#include "POOLCore/Token.h"
//#include "FileCatalog/URIParser.h"
//#include "FileCatalog/FCSystemTools.h"
//#include "FileCatalog/IFileCatalog.h"
//#include "StorageSvc/DbType.h"
//#include "PersistencySvc/DatabaseConnectionPolicy.h"
//#include "PersistencySvc/ISession.h"
//#include "PersistencySvc/ITransaction.h"
//#include "PersistencySvc/IDatabase.h"
//#include "PersistencySvc/Placement.h"
//#include "DataSvc/DataSvcFactory.h"
//#include "DataSvc/IDataSvc.h"
//#include "DataSvc/ICacheSvc.h"

#include <iostream>
#include <cmath>

HcalTestHistoManager::HcalTestHistoManager(const std::string & file) // : 
//  svc(pool::DataSvcFactory::instance(&lcat)),
//  placeH(file, pool::DatabaseSpecification::PFN, "HcalTestAnalysis", 
//	 ROOT::Reflex::Type(), pool::ROOTTREE_StorageType.type()), h(*svc) {
  {  
    //  pool::URIParser p("file:HcalTestHistoCatalog.cat");
    //p.parse();
 
    //lcat.setWriteCatalog(p.contactstring());
 
    //lcat.connect();
    //  lcat.start();

  // Define the policy for the implicit file handling
  //pool::DatabaseConnectionPolicy policy;
  //policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  //policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
  ///// policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  //svc->session().setDefaultConnectionPolicy(policy);
  
  edm::LogInfo("HcalSim") << "HcalTestHistoManager:===>>>  Book user"
			  << " Histograms and Root tree";
}

HcalTestHistoManager::~HcalTestHistoManager() {

  edm::LogInfo("HcalSim") << "============================================="
			  << "==============\n"
			  << "=== HcalTestHistoManager: Start writing user "
			  << "histograms ===";

  //svc->transaction().commit();
  //svc->session().disconnectAll();
  //edm::LogInfo("HcalSim") << "=== HcalTestHistoManager: cache size at end " 
  //			  << svc->cacheSvc().cacheSize();
  //lcat.commit();
 
  edm::LogInfo("HcalSim") << "=== HcalTestHistoManager: End   writing user "
			  << "histograms ===\n"
			  << "============================================="
			  << "==============";
}

void HcalTestHistoManager::fillTree(HcalTestHistoClass *  histos) {

  //svc->transaction().start(pool::ITransaction::UPDATE);
  LogDebug("HcalSim") << "HcalTestHistoManager: tree pointer = " << histos;
  //LogDebug("HcalSim") << "HcalTestHistoManager: cache size before assign " 
  //		      << svc->cacheSvc().cacheSize();

  //h = histos;
  //h.markWrite(placeH);
  //svc->transaction().commitAndHold();
}
