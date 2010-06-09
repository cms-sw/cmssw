//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 * $Id$
 *
 * author: Ph Gras. June, 2010
 */

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSRCondTools.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"

#include <string>
#include <fstream>

using namespace std;

EcalSRCondTools::EcalSRCondTools(const edm::ParameterSet& ps):ps_(ps){
}


EcalSRCondTools::~EcalSRCondTools(){
}

void
EcalSRCondTools::analyze(const edm::Event& event, const edm::EventSetup& es){
  static bool done = false;
  if(done) return;
  EcalSRSettings* sr = new EcalSRSettings;

  string mode = ps_.getParameter<string>("mode");
  bool iomode;
  const bool iomode_read = false;
  const bool iomode_write = true;

  if(mode == "online_config" || mode == "combine_config"){
    iomode =iomode_write;
    string fname = ps_.getParameter<string>("onlineSrpConfigFile");
    ifstream f(fname.c_str());
    if(!f.good()){
      throw cms::Exception("EcalSRCondTools") << "Failed to open file " << fname;
    }
    sr->importSrpConfigFile(f, true);
  }

  if(mode=="python_config" || mode=="combine_config"){
    iomode = iomode_write;
    sr->importParameterSet(ps_);
  }

  if(mode=="read"){
    iomode = iomode_read;
  }

  if(!(mode=="python_config" || mode == "online_config" || mode == "combine_config" || (mode=="read"))){
    throw cms::Exception("Config") << "Invalid value," << mode << ",  for parameter mode. "
                                   << "Valid values: online_config, python_config, combine_config, read";
  }

  if(iomode==iomode_write){
    sr->bxGlobalOffset_ = ps_.getParameter<int>("bxGlobalOffset");
    sr->automaticSrpSelect_ = ps_.getParameter<int>("automaticSrpSelect");
    sr->automaticMasks_ = ps_.getParameter<int>("automaticMasks");
    
    edm::Service<cond::service::PoolDBOutputService> db;
    if( !db.isAvailable() ){
      throw cms::Exception("CondDBAccess") << "Failed to connect to PoolDBOutputService\n";
    }
    //fillup DB
    //create new infinite IOV
    cond::Time_t firstSinceTime = db->beginOfTime();
    db->writeOne(sr,firstSinceTime,"EcalSRSettingsRcd");
    done = true;
  } else {//read mode
    edm::ESHandle<EcalSRSettings> hSr;
    es.get<EcalSRSettingsRcd>().get(hSr);
    const EcalSRSettings* sr = hSr.product();
    cout << "ECAL Seletive readout settings:\n";
    cout << *sr << "\n" << endl;
  }
}
