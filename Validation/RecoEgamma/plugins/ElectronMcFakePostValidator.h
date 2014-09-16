
#ifndef Validation_RecoEgamma_ElectronMcFakePostValidator_h
#define Validation_RecoEgamma_ElectronMcFakePostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"

class ElectronMcFakePostValidator : public ElectronDqmHarvesterBase
 {
  public:
    explicit ElectronMcFakePostValidator( const edm::ParameterSet & conf ) ; 
    virtual ~ElectronMcFakePostValidator() ;
    virtual void book() ;
    virtual void finalize2() ;
    virtual void finalize( DQMStore::IBooker & iBooker ) ; // , DQMStore::IGetter & iGetter

  private:
    // histos limits and binning

    bool set_EfficiencyFlag ;

    // histos
    MonitorElement *h1_ele_xOverX0VsEta ;
	
 } ;

#endif



