#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string.h>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fstream>

EcalElectronicsSim::EcalElectronicsSim( const EcalSimParameterMap* parameterMap      , 
					EcalCoder*                 coder             , 
					bool                       applyConstantTerm , 
					double                     rmsConstantTerm     ) :
   m_simMap             ( parameterMap ) ,
   m_theCoder           ( coder        ) ,
   m_gaussQDistribution ( 0            )
{
   edm::Service<edm::RandomNumberGenerator> rng;
 
   if( applyConstantTerm )
   {
      if ( !rng.isAvailable() ) 
      {
	 throw cms::Exception("Configuration")
	    << "EcalElectroncSim requires the RandomNumberGeneratorService\n"
	    "which is not present in the configuration file.  You must add the service\n"
	    "in the configuration file or remove the modules that require it.";
      }

      double thisCT = rmsConstantTerm ;
      m_gaussQDistribution = new CLHEP::RandGaussQ( rng->getEngine(), 1.0, thisCT ) ;
   }
}

EcalElectronicsSim::~EcalElectronicsSim()
{  
   delete m_gaussQDistribution ;
}

void 
EcalElectronicsSim::analogToDigital( EcalElectronicsSim::EcalSamples& clf , 
				     EcalDataFrame&                   df    ) const 
{

   //PG input signal is in pe.  Converted in GeV
   amplify( clf ) ;

   m_theCoder->analogToDigital( clf, df ) ;
}

void 
EcalElectronicsSim::amplify( EcalElectronicsSim::EcalSamples& clf ) const 
{
   const double fac ( m_simMap->simParameters( clf.id() ).photoelectronsToAnalog() ) ;
   if( 0 != m_gaussQDistribution ) 
   {
      clf *= fac*m_gaussQDistribution->fire() ;
   }
   else
   {
      clf *= fac ;
   }
}




