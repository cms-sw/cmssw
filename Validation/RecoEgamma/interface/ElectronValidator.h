
#ifndef Validation_RecoEgamma_ElectronValidator_h
#define Validation_RecoEgamma_ElectronValidator_h

class DQMStore;
class MonitorElement;

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Rtypes.h>

class ElectronValidator : public edm::EDAnalyzer
 {

  protected:

    explicit ElectronValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronValidator() ;

    void prepareStore() ;
    void setStoreFolder( const std::string & path ) ;
    void saveStore( const std::string & filename ) ;

    MonitorElement * bookH1
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events" ) ;

    MonitorElement * bookH1withSumw2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events" ) ;

    MonitorElement * bookH2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
     const std::string & titleX ="", const std::string & titleY ="" ) ;

    MonitorElement * bookH2withSumw2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="" ) ;

    MonitorElement * bookP1
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
                 double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="" ) ;

    MonitorElement * bookH1andDivide
     ( const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="", bool print =false ) ;

    MonitorElement * bookH2andDivide
     ( const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="", bool print =false ) ;

    MonitorElement * profileX
     ( const std::string & name, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( const std::string & name, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

  private:

    DQMStore * store_ ;

 } ;

#endif



