#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h" 
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
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
   m_apdShape ( apdShape ) ,
   m_timeOffVec ( kNOffsets, apdParameters()->timeOffset() )
{
   for( unsigned int i ( 0 ) ; i != kNOffsets ; ++i )
   {
      m_timeOffVec[ i ] +=
	 ranGauss()->fire( 0 , apdParameters()->timeOffWidth() ) ;
   }
}

EBHitResponse::~EBHitResponse()
{
}


const APDSimParameters*
EBHitResponse::apdParameters() const
{
   assert ( 0 != m_apdPars ) ;
   return m_apdPars ;
}

const CaloVShape*
EBHitResponse::apdShape() const
{
   assert( 0 != m_apdShape ) ;
   return m_apdShape ;
}

void
EBHitResponse::putAnalogSignal( const PCaloHit& hit )
{
   const unsigned int depth ( hit.depth() ) ;
   if( !m_apdOnly &&
       0 == depth    )
   {
      CaloHitResponse::putAnalogSignal( hit ) ;
   }
   else
   {
      if( 0 != depth                           &&
	  ( apdParameters()->addToBarrel() ||
	    m_apdOnly                        )    ) // can digitize apd
      {
	 const DetId detId  ( hit.id() ) ;
	 CaloSamples& result ( *findSignal( detId ) );

//	 edm::LogError( "EBHitResponse" )<<"---APD SimHit found for "
/*	 std::cout<<"---APD SimHit found for "
		  << EBDetId( detId ) 
		  <<", depth="<< depth 
		  <<std::endl ;*/

	 const double signal ( apdSignalAmplitude( hit ) ) ;
	    
	 const CaloSimParameters& parameters ( *params( detId ) ) ;

	 const double jitter ( hit.time() - timeOfFlight( detId ) ) ;

	 const double tzero ( apdShape()->timeToRise()
			      - jitter
			      - offsets()[ EBDetId( detId ).denseIndex()%kNOffsets ]
			      - BUNCHSPACE*( parameters.binOfMaximum()
					     - phaseShift()            ) ) ;
	 double binTime ( tzero ) ;

	 for( int bin ( 0 ) ; bin != result.size(); ++bin )
	 {
	    result[bin] += (*apdShape())(binTime)*signal;
	    binTime += BUNCHSPACE;
	 }
      }
   } 
}

double 
EBHitResponse::apdSignalAmplitude( const PCaloHit& hit ) const 
{
   assert( 1 == hit.depth() ||
	   2 == hit.depth()    ) ;

   double npe ( hit.energy()*( 2 == hit.depth() ?
			       apdParameters()->simToPELow() :
			       apdParameters()->simToPEHigh() ) ) ;
			       
   // do we need to doPoisson statistics for the photoelectrons?
   if( apdParameters()->doPEStats() &&
       !m_apdOnly                      ) npe = ranPois()->fire( npe ) ;

   assert( 0 != m_intercal ) ;
   double fac ( 1 ) ;
   findIntercalibConstant( hit.id(), fac ) ;

   npe *= fac ;
//   edm::LogError( "EBHitResponse" ) << "--- # photoelectrons for "
/*   std::cout << "--- # photoelectrons for "
	     << EBDetId( hit.id() ) 
	     <<" is " << npe //;
	     <<std::endl ;*/

   return npe ;
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
      edm::LogError( "EBHitResponse" ) << 
	 "No intercal constant defined for EBHitResponse" ;
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
