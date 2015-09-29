
#include "Validation/RecoEgamma/plugins/ElectronMcMiniAODSignalPostValidator.h" 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcMiniAODSignalPostValidator::ElectronMcMiniAODSignalPostValidator( const edm::ParameterSet & conf )
 : ElectronDqmHarvesterBase(conf)
 {
  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");
 }

ElectronMcMiniAODSignalPostValidator::~ElectronMcMiniAODSignalPostValidator()
 {}

void ElectronMcMiniAODSignalPostValidator::finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter )
 {

  setBookIndex(-1) ;
  setBookPrefix("h_ele") ;
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag( set_StatOverflowFlag ) ;
  
  // profiles from 2D histos
  profileX(iBooker, iGetter, "PoPtrueVsEta","mean ele momentum / gen momentum vs eta","#eta","<P/P_{gen}>");
  profileX(iBooker, iGetter, "sigmaIetaIetaVsPt","SigmaIetaIeta vs pt","p_{T} (GeV/c)","SigmaIetaIeta");
/**/



}


