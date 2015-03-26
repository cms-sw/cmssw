#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CLHEP/Random/RandGaussQ.h"

#include <iostream>

ESElectronicsSimFast::ESElectronicsSimFast( bool addNoise , bool PreMix1 ) :
   m_addNoise ( addNoise ) ,
   m_PreMix1  ( PreMix1  ) ,
   m_MIPToGeV (        0 ) ,
   m_peds     (        0 ) ,
   m_mips     (        0 )
{
   // Preshower "Fast" Electronics Simulation
   // gain = 1 : low gain for data taking 
   // gain = 2 : high gain for calibration and low energy runs
   // For 300(310/320) um Si, the MIP is 78.47(81.08/83.7) keV
}

ESElectronicsSimFast::~ESElectronicsSimFast()
{
}

void 
ESElectronicsSimFast::setPedestals( const ESPedestals* peds ) 
{
   m_peds = peds ;
} 

void 
ESElectronicsSimFast::setMIPs( const ESIntercalibConstants* mips ) 
{
   m_mips = mips ;
}

void 
ESElectronicsSimFast::setMIPToGeV( double MIPToGeV )
{
   m_MIPToGeV = MIPToGeV ;
}

void 
ESElectronicsSimFast::analogToDigital( CLHEP::HepRandomEngine* engine,
                                       ESSamples&   cs,
			               ESDataFrame& df,
			               bool         isNoise ) const
{
   assert( 0 != m_peds &&
	   0 != m_mips &&
	   0 < m_MIPToGeV ) ; // sanity check

   df.setSize( cs.size() ) ;

   const DetId id ( cs.id() ) ;
   ESPedestals::const_iterator it_ped ( m_peds->find( id ) ) ;
   ESIntercalibConstantMap::const_iterator it_mip (
      isNoise ? m_mips->getMap().end() : m_mips->getMap().find( id ) ) ;

   const double baseline ( (double) it_ped->getMean() ) ;
   const double sigma    ( isNoise ? 0. : (double) it_ped->getRms() ) ;
   const double MIPADC   ( isNoise ? 0. : (double) (*it_mip) ) ;
   const double ADCGeV   ( isNoise ? 1. : MIPADC/m_MIPToGeV ) ;

   int adc = 0 ;
//   std::cout<<"   **Id="<<ESDetId(df.id())<<", size="<<df.size();
   for( unsigned int i ( 0 ) ; i != cs.size(); ++i ) 
   {
      const double noi ( isNoise || (!m_addNoise) ? 0 :
			 sigma*CLHEP::RandGaussQ::shoot(engine, 0, 1) ) ;
      double signal;

      if(!m_PreMix1) signal = cs[i]*ADCGeV + noi + baseline ;
      else signal = cs[i]*ADCGeV ;

      if( 0 <= signal )
      { 
	 signal += 0.5 ;
      }
      else
      {
	 signal -= 0.5 ;
      }
    
      adc = int( signal ) ;

      if(!m_PreMix1) assert( 0 < adc ) ;

      if( 0.5 < signal - adc ) ++adc ;

      if( MAXADC < adc )
      {
	 adc = MAXADC ;
      }
      else
      {
	 if( MINADC > adc ) adc = MINADC ;
      }

      df.setSample( i, ESSample( adc ) ) ;
//      std::cout<<", "<<df[i];
   }
//   std::cout<<std::endl ;
}
