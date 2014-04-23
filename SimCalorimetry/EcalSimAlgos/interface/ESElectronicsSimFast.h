#ifndef EcalSimAlgos_ESElectronicsSimFast_h
#define EcalSimAlgos_ESElectronicsSimFast_h 1

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"

#include <vector>

namespace CLHEP {
   class RandGaussQ ; } 

class ESElectronicsSimFast
{
   public:

      typedef CaloTSamples<float,3> ESSamples ;

      enum { MAXADC = 4095,
	     MINADC =    0 } ;
  
      ESElectronicsSimFast( bool addNoise , bool PreMix1) ;
      ~ESElectronicsSimFast() ;

      void setPedestals( const ESPedestals* peds ) ;

      void setMIPs( const ESIntercalibConstants* mips ) ;

      void setMIPToGeV( double MIPToGeV ) ;

      void analogToDigital( ESSamples&   cs , 
			    ESDataFrame& df ,
			    bool         isNoise = false ) const ;

      void newEvent() {}


   private :

      bool m_addNoise ;

      bool m_PreMix1;

      double m_MIPToGeV ;

      const ESPedestals* m_peds ;

      const ESIntercalibConstants* m_mips ;

      CLHEP::RandGaussQ* m_ranGau ;
} ;


#endif
