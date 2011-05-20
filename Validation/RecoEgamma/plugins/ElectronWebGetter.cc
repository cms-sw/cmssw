
#include "Validation/RecoEgamma/plugins/ElectronWebGetter.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ElectronWebGetter::ElectronWebGetter( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {}

ElectronWebGetter::~ElectronWebGetter()
 {}

void ElectronWebGetter::finalize()
 {
  remove_other_dirs() ;
 }


