
#include "Validation/RecoEgamma/plugins/ElectronWebGetter.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ElectronWebGetter::ElectronWebGetter( const edm::ParameterSet & conf )
 : ElectronDqmHarvesterBase(conf)
 {}

ElectronWebGetter::~ElectronWebGetter()
 {}

void ElectronWebGetter::finalize(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter )
 {
  remove_other_dirs(iBooker, iGetter ) ;
 }


