// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestHistoManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id$
//

// system include files
#include <iostream>
#include <cmath>

// user include files
#include "SimG4CMS/Forward/interface/TotemTestHistoManager.h"
#include "SimG4CMS/Forward/interface/TotemTestHistoClass.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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


//
// constructors and destructor
//

TotemTestHistoManager::TotemTestHistoManager(const std::string & file) : 
  svc(pool::DataSvcFactory::instance(&lcat)),
  placeH(file, pool::DatabaseSpecification::PFN, "TotemTestAnalysis", 
	 ROOT::Reflex::Type(), pool::ROOTTREE_StorageType.type()), h(*svc) {
  
  pool::URIParser p("file:TotemTestHistoCatalog.cat");
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
  
  edm::LogInfo("ForwardSim") << "TotemTestHistoManager:===>>>  Book user"
			     << " Histograms and Root tree";
}

TotemTestHistoManager::~TotemTestHistoManager() {

  edm::LogInfo("ForwardSim") << "============================================="
			     << "==============\n"
			     << "=== TotemTestHistoManager: Start writing user"
			     << " histograms ===";

  svc->transaction().commit();
  svc->session().disconnectAll();
  edm::LogInfo("ForwardSim") << "=== TotemTestHistoManager: cache size at end" 
			     << " " << svc->cacheSvc().cacheSize();
  lcat.commit();
 
  edm::LogInfo("ForwardSim") << "=== TotemTestHistoManager: End   writing user"
			     << " histograms ===\n"
			     << "============================================="
			     << "==============";
}

void TotemTestHistoManager::fillTree(TotemTestHistoClass *  histos) {

  svc->transaction().start(pool::ITransaction::UPDATE);
  LogDebug("ForwardSim") << "TotemTestHistoManager: tree pointer = " << histos;
  LogDebug("ForwardSim") << "TotemTestHistoManager: cache size before assign " 
			 << svc->cacheSvc().cacheSize();

  h = histos;
  h.markWrite(placeH);
  svc->transaction().commitAndHold();
}
