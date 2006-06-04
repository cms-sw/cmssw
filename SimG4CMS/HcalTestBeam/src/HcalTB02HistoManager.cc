// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HistoManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Sun May 21 10:14:34 CEST 2006
// $Id$
//

// system include files
#include <iostream>
#include <cmath>

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HistoManager.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HistoClass.h"
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

HcalTB02HistoManager::HcalTB02HistoManager(const std::string & file) : 
  svc(pool::DataSvcFactory::instance(&lcat)),
  placeH(file, pool::DatabaseSpecification::PFN, "HcalTB02Analysis", 
	 ROOT::Reflex::Type(), pool::ROOTTREE_StorageType.type()), h(*svc) {
  
  pool::URIParser p("file:HcalTB02HistoCatalog.cat");
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
  
  edm::LogInfo("HcalTBSim") << "HcalTB02HistoManager:===>>>  Book user"
			     << " Histograms and Root tree";
}

HcalTB02HistoManager::~HcalTB02HistoManager() {

  edm::LogInfo("HcalTBSim") << "============================================="
			    << "==============\n"
			    << "=== HcalTB02HistoManager: Start writing user"
			    << " histograms ===";

  svc->transaction().commit();
  svc->session().disconnectAll();
  edm::LogInfo("HcalTBSim") << "=== HcalTB02HistoManager: cache size at end" 
			    << " " << svc->cacheSvc().cacheSize();
  lcat.commit();
 
  edm::LogInfo("HcalTBSim") << "=== HcalTB02HistoManager: End   writing user"
			    << " histograms ===\n"
			    << "============================================="
			    << "==============";
}

//
// member functions
//

void HcalTB02HistoManager::fillTree(HcalTB02HistoClass *  histos) {

  svc->transaction().start(pool::ITransaction::UPDATE);
  LogDebug("HcalTBSim") << "HcalTB02HistoManager: tree pointer = " << histos
			<< "\nHcalTB02HistoManager: cache size before assign " 
			<< svc->cacheSvc().cacheSize();

  h = histos;
  h.markWrite(placeH);
  svc->transaction().commitAndHold();
}
