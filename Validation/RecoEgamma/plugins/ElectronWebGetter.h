
#ifndef Validation_RecoEgamma_ElectronWget_h
#define Validation_RecoEgamma_ElectronWget_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"

class ElectronWebGetter : public ElectronDqmHarvesterBase
 {
  public:
    explicit ElectronWebGetter( const edm::ParameterSet & conf ) ;
    virtual ~ElectronWebGetter() ;
    virtual void finalize(DQMStore::IGetter & iGetter) ;
 } ;

#endif



