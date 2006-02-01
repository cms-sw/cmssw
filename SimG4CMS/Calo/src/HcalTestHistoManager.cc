///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.cc
// Description: Histogram managing class in HcalTestAnalysis (HcalTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoManager.h"
#include "SimG4CMS/Calo/interface/HcalTestHistoClass.h"

//#include "Reflection/Class.h"
#include "PluginManager/PluginManager.h"
#include "POOLCore/Token.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/ICacheSvc.h"

#include <iostream>
#include <cmath>

HcalTestHistoManager::HcalTestHistoManager(int iv, const std::string & file) : 
  verbosity(iv), svc(pool::DataSvcFactory::instance(&lcat)),
  placeH(file, pool::DatabaseSpecification::PFN, "HcalTestAnalysis", 
	  ROOT::Reflex::Type(), pool::ROOTTREE_StorageType.type()), h(*svc) {
  
  pool::URIParser p("file:HcalTestHistoCatalog.cat");
  p.parse();
 
  lcat.setWriteCatalog(p.contactstring());
 
  lcat.connect();
  lcat.start();

  // Define the policy for the implicit file handling
  pool::DatabaseConnectionPolicy policy;
  policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
  policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
  // policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
  svc->session().setDefaultConnectionPolicy(policy);
  
  if (verbosity > 0)
    std::cout << std::endl << "===>>>  Start booking user Root tree" 
	      << std::endl;

  if (verbosity > 0) {
    std::cout << std::endl << "===>>> Done booking user histograms & Ntuples " 
	      << std::endl;
  }
}

HcalTestHistoManager::~HcalTestHistoManager() {

  if (verbosity > 0) 
    std::cout << "===========================================================" 
	      << std::endl
	      << "=== HcalTestHistoManager: Start writing user histograms ===" 
	      << std::endl;

  svc->transaction().commit();
  svc->session().disconnectAll();
  if (verbosity > 0) 
    std::cout << "=== HcalTestHistoManager: cache size at end " 
	      << svc->cacheSvc().cacheSize() << std::endl;
  lcat.commit();
 
  if (verbosity > 0) 
    std::cout << std::endl 
	      << "=== HcalTestHistoManager: End   writing user histograms ==="
	      << std::endl
	      << "===========================================================" 
	      << std::endl;
}

void HcalTestHistoManager::fillTree(HcalTestHistoClass *  histos) {

   svc->transaction().start(pool::ITransaction::UPDATE);
   if (verbosity > 1) {
     std::cout << " tree pointer = " << histos << std::endl;
     std::cout << "cache size before assign " << svc->cacheSvc().cacheSize() 
	       << std::endl;
  }

  h = histos;
  h.markWrite(placeH);
  svc->transaction().commitAndHold();

}
