#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h" 
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


EBHitResponse::EBHitResponse( const CaloVSimParameterMap* parameterMap , 
			      const CaloVShape*           shape        ,
			      bool                        apdOnly      ,
			      const APDSimParameters*     apdPars  = 0 , 
			      const CaloVShape*           apdShape = 0   ) :

   CaloHitResponse( parameterMap,
		    shape         ),

   m_apdOnly  ( apdOnly  ) ,
   m_apdPars  ( apdPars  ) ,
   m_apdShape ( apdShape ) 
{
}

EBHitResponse::~EBHitResponse()
{
}


const APDSimParameters*
EBHitResponse::apdParameters() const
{
   return m_apdPars ;
}

const CaloVShape*
EBHitResponse::apdShape() const
{
   return m_apdShape ;
}

CaloSamples 
EBHitResponse::makeAnalogSignal( const PCaloHit& hit ) const 
{
   const unsigned int depth ( hit.depth() ) ;
   if( !m_apdOnly &&
       0 == depth    )
   {
      return CaloHitResponse::makeAnalogSignal( hit ) ;
   }
   else
   {
      const DetId detId  ( hit.id() ) ;
      CaloSamples result ( makeBlankSignal( detId ) );

      if( 0 != depth ) std::cout<<"........found an apd simhit for "
				<<EBDetId(detId) <<", depth="<<depth<<std::endl;
      if( 0 != depth            &&
	  0 != apdParameters()  &&
	  0 != apdShape()       &&
	  apdParameters()->addToBarrel() ) // can digitize apd
      {
	 const double signal ( apdSignalAmplitude( hit ) ) ;
	    
	 const CaloSimParameters& parameters ( theParameterMap->simParameters( detId ) ) ;

	 double jitter = hit.time() - timeOfFlight( detId ) ;

	 const double tzero ( theShape->timeToRise()
			      - jitter
			      - apdParameters()->timeOffset() 
			      - BUNCHSPACE*( parameters.binOfMaximum()
					     - thePhaseShift_          ) ) ;
	 double binTime ( tzero ) ;

	 for( int bin ( 0 ) ; bin != result.size(); ++bin )
	 {
	    result[bin] += (*apdShape())(binTime)*signal;
	    binTime += BUNCHSPACE;
	 }
      }
      return result ;
   } 
}


double 
EBHitResponse::apdSignalAmplitude( const PCaloHit& hit ) const 
{
   if( 0 == theRandPoisson )
   {
      edm::Service<edm::RandomNumberGenerator> rng;
      if ( ! rng.isAvailable()) {
	 throw cms::Exception("Configuration")
	    << "EBHitResponse requires the RandomNumberGeneratorService\n"
	    "which is not present in the configuration file.  You must add the service\n"
	    "in the configuration file or remove the modules that require it.";
      }
      theRandPoisson = new CLHEP::RandPoissonQ(rng->getEngine());
   }

   assert( 1 == hit.depth() ||
	   2 == hit.depth()    ) ;

   double npe = hit.energy()*( 2 == hit.depth() ?
			       apdParameters()->simToPELow() :
			       apdParameters()->simToPEHigh() ) ;
			       
   // do we need to doPoisson statistics for the photoelectrons?
   if( apdParameters()->doPEStats() )
   {
      npe = theRandPoisson->fire( npe ) ;
   }

   if( 0 != m_intercal ) 
   {
      double fac ( 1 ) ;
      findIntercalibConstant( hit.id(), fac ) ;
      npe *= fac ;
      std::cout<<".... number of photoelectrons for "
	       <<EBDetId(hit.id())<<" after INTERCAL FACTOR is " << npe<<std::endl ;
   }

   return npe;
}

void 
EBHitResponse::setIntercal( const EcalIntercalibConstantsMC* ical )
{
   m_intercal = ical ;
}

void 
EBHitResponse::findIntercalibConstant( const DetId& detId, 
				       double&      icalconst ) const
{
   EcalIntercalibConstantMC thisconst ( 1. ) ;

   if( 0 == m_intercal )
   {
      edm::LogError("EBHitResponse") <<"No intercal constant defined for EBHitResponse" ;
   }
   else
   {
      const EcalIntercalibConstantMCMap&          icalMap ( m_intercal->getMap()  ) ;
      EcalIntercalibConstantMCMap::const_iterator icalit  ( icalMap.find( detId ) ) ;
      if( icalit != icalMap.end() )
      {
	 thisconst = *icalit ;
	 if ( thisconst == 0. ) thisconst = 1. ; 
      } 
      else
      {
	 edm::LogError("EBHitResponse") << "No intercalib const found for xtal " 
					<< detId.rawId() 
					<< "! something wrong with EcalIntercalibConstants in your DB? ";
      }
   }
   icalconst = thisconst ;
}
