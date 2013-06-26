#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h" 
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/isFinite.h"


EBHitResponse::EBHitResponse( const CaloVSimParameterMap* parameterMap , 
			      const CaloVShape*           shape        ,
			      bool                        apdOnly      ,
			      const APDSimParameters*     apdPars  = 0 , 
			      const CaloVShape*           apdShape = 0   ) :

   EcalHitResponse( parameterMap, shape ) ,

   m_apdOnly  ( apdOnly  ) ,
   m_apdPars  ( apdPars  ) ,
   m_apdShape ( apdShape ) ,
   m_timeOffVec ( kNOffsets, apdParameters()->timeOffset() ) ,
   pcub ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[0] ) ,
   pqua ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[1] ) ,
   plin ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[2] ) ,
   pcon ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[3] ) ,
   pelo ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[4] ) ,
   pehi ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[5] ) ,
   pasy ( 0 == apdPars ? 0 : apdParameters()->nonlParms()[6] ) ,
   pext ( 0 == apdPars ? 0 : nonlFunc1( pelo ) ) ,
   poff ( 0 == apdPars ? 0 : nonlFunc1( pehi ) ) ,
   pfac ( 0 == apdPars ? 0 : ( pasy - poff )*2./M_PI ) 
{
   for( unsigned int i ( 0 ) ; i != kNOffsets ; ++i )
   {
      m_timeOffVec[ i ] +=
	 ranGauss()->fire( 0 , apdParameters()->timeOffWidth() ) ;
   }

   const EBDetId detId ( EBDetId::detIdFromDenseIndex( 0 ) ) ;
   const CaloSimParameters& parameters ( parameterMap->simParameters( detId ) ) ;

   const unsigned int rSize ( parameters.readoutFrameSize() ) ;
   const unsigned int nPre  ( parameters.binOfMaximum() - 1 ) ;

   const unsigned int size ( EBDetId::kSizeForDenseIndexing ) ;

   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSam.emplace_back(CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
		    rSize, nPre );
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
EBHitResponse::putAPDSignal( const DetId& detId  ,
			     double       npe    ,
			     double       time     )
{
   const CaloSimParameters& parameters ( *params( detId ) ) ;

   const double energyFac ( 1./parameters.simHitToPhotoelectrons( detId ) ) ;

//   std::cout<<"******** Input APD Npe="<<npe<<", Efactor="<<energyFac
//	    <<", Energy="<<npe*energyFac
//	    <<", nonlFunc="<<nonlFunc( npe*energyFac )<<std::endl ;

   const double signal ( npe*nonlFunc( npe*energyFac ) ) ;

   const double jitter ( time - timeOfFlight( detId ) ) ;

   const double tzero ( apdShape()->timeToRise()
			- jitter
			- offsets()[ EBDetId( detId ).denseIndex()%kNOffsets ]
			- BUNCHSPACE*( parameters.binOfMaximum()
				       - phaseShift()            ) ) ;

   double binTime ( tzero ) ;

   EcalSamples& result ( *findSignal( detId ) );

   for( unsigned int bin ( 0 ) ; bin != result.size(); ++bin )
   {
      result[bin] += (*apdShape())(binTime)*signal ;
      binTime += BUNCHSPACE ;
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

   // do we need to do Poisson statistics for the photoelectrons?
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

void 
EBHitResponse::initializeHits() {
   if( 0 != index().size() ) blankOutUsedSamples() ;

   const unsigned int bSize ( EBDetId::kSizeForDenseIndexing ) ;

   if( 0 == m_apdNpeVec.size() )
   {
      m_apdNpeVec  = std::vector<double>( bSize, (double)0.0 ) ;
      m_apdTimeVec = std::vector<double>( bSize, (double)0.0 ) ;
   }
}

void 
EBHitResponse::finalizeHits() {
   const unsigned int bSize ( EBDetId::kSizeForDenseIndexing ) ;
   if( apdParameters()->addToBarrel() ||
       m_apdOnly                         )
   {
      for( unsigned int i ( 0 ) ; i != bSize ; ++i )
      {
         if( 0 < m_apdNpeVec[i] )
         {
            putAPDSignal( EBDetId::detIdFromDenseIndex( i ),
                          m_apdNpeVec[i] ,
                          m_apdTimeVec[i]                    ) ;

            // now zero out for next time
            m_apdNpeVec[i] = 0. ;
            m_apdTimeVec[i] = 0. ;
         }
      }
   }
}

void 
EBHitResponse::add( const PCaloHit& hit ) 
{
  if (!edm::isNotFinite( hit.time() ) && ( 0 == hitFilter() || hitFilter()->accepts( hit ) ) ) {
     if( 0 == hit.depth() ) // for now take only nonAPD hits
     {
        if( !m_apdOnly ) putAnalogSignal( hit ) ;
     }
     else // APD hits here
     {
        if( apdParameters()->addToBarrel() ||
            m_apdOnly                         )
        {
           const unsigned int icell ( EBDetId( hit.id() ).denseIndex() ) ;
           m_apdNpeVec[ icell ] += apdSignalAmplitude( hit ) ;
           if( 0 == m_apdTimeVec[ icell ] ) m_apdTimeVec[ icell ] = hit.time() ;
        }
     }
  }
}

void 
EBHitResponse::run( MixCollection<PCaloHit>& hits ) 
{
   if( 0 != index().size() ) blankOutUsedSamples() ;

   const unsigned int bSize ( EBDetId::kSizeForDenseIndexing ) ;

   if( 0 == m_apdNpeVec.size() ) 
   {
      m_apdNpeVec  = std::vector<double>( bSize, (double)0.0 ) ;
      m_apdTimeVec = std::vector<double>( bSize, (double)0.0 ) ;
   }

   for( MixCollection<PCaloHit>::MixItr hitItr ( hits.begin() ) ;
	hitItr != hits.end() ; ++hitItr )
   {
      const PCaloHit& hit ( *hitItr ) ;
      const int bunch ( hitItr.bunch() ) ;
      if( minBunch() <= bunch  &&
	  maxBunch() >= bunch  &&
	  !edm::isNotFinite( hit.time() ) &&
	  ( 0 == hitFilter() ||
	    hitFilter()->accepts( hit ) ) )
      { 
	 if( 0 == hit.depth() ) // for now take only nonAPD hits
	 {
	    if( !m_apdOnly ) putAnalogSignal( hit ) ;
	 }
	 else // APD hits here
	 {
	    if( apdParameters()->addToBarrel() ||
		m_apdOnly                         )
	    {
	       const unsigned int icell ( EBDetId( hit.id() ).denseIndex() ) ;
	       m_apdNpeVec[ icell ] += apdSignalAmplitude( hit ) ;
	       if( 0 == m_apdTimeVec[ icell ] ) m_apdTimeVec[ icell ] = hit.time() ;
	    }
	 }
      }
   }

   if( apdParameters()->addToBarrel() ||
       m_apdOnly                         )
   {
      for( unsigned int i ( 0 ) ; i != bSize ; ++i )
      {
	 if( 0 < m_apdNpeVec[i] )
	 {
	    putAPDSignal( EBDetId::detIdFromDenseIndex( i ),
			  m_apdNpeVec[i] ,
			  m_apdTimeVec[i]                    ) ;

	    // now zero out for next time
	    m_apdNpeVec[i] = 0. ;
	    m_apdTimeVec[i] = 0. ;
	 }
      }
   }
}

unsigned int
EBHitResponse::samplesSize() const
{
   return m_vSam.size() ;
}

unsigned int
EBHitResponse::samplesSizeAll() const
{
   return m_vSam.size() ;
}

const EcalHitResponse::EcalSamples* 
EBHitResponse::operator[]( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EBHitResponse::operator[]( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EBHitResponse::vSam( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EBHitResponse::vSamAll( unsigned int i )
{
   return &m_vSam[ i ] ;
}

const EcalHitResponse::EcalSamples* 
EBHitResponse::vSamAll( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}
