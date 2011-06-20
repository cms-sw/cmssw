#ifndef EcalSimAlgos_ESElectronicsSimFast_h
#define EcalSimAlgos_ESElectronicsSimFast_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
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
  
      enum { MAXADC = 4095,
	     MINADC =    0 } ;
  
      ESElectronicsSimFast( bool addNoise ) ;
      ~ESElectronicsSimFast() ;

      void setPedestals( const ESPedestals* peds ) ;

      void setMIPs( const ESIntercalibConstants* mips ) ;

      void setMIPToGeV( double MIPToGeV ) ;

      virtual void analogToDigital( const CaloSamples& cs, 
				    ESDataFrame&       df,
				    bool               isNoise ) const;

   private :

      bool m_addNoise ;

      double m_MIPToGeV ;

      const ESPedestals* m_peds ;

      const ESIntercalibConstants* m_mips ;

      CLHEP::RandGaussQ* m_ranGau ;
} ;


#endif
