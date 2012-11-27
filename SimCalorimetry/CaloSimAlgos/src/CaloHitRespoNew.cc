#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitRespoNew.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVPECorrection.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 
#include<iostream>



CaloHitRespoNew::CaloHitRespoNew( const CaloVSimParameterMap* parameterMap ,
				  const CaloVShape*           shape        ,
				  DetId                       detId         ) :
   m_parameterMap  ( parameterMap ) ,
   m_shape         ( shape        ) ,
   m_hitCorrection ( 0            ) ,
   m_PECorrection  ( 0            ) ,
   m_hitFilter     ( 0            ) ,
   m_geometry      ( 0            ) ,
   m_RandPoisson   ( 0            ) ,
   m_RandGauss     ( 0            ) ,
   m_minBunch      ( -10          ) ,
   m_maxBunch      (  10          ) ,
   m_phaseShift    ( 1            ) 
{
   setupSamples( detId ) ;
}

CaloHitRespoNew::~CaloHitRespoNew()
{
   delete m_RandPoisson ;
   delete m_RandGauss   ;
}

CLHEP::RandPoissonQ* 
CaloHitRespoNew::ranPois() const
{
   if( 0 == m_RandPoisson )
   {
      edm::Service<edm::RandomNumberGenerator> rng ;
      if ( !rng.isAvailable() ) 
      {
	 throw cms::Exception("Configuration")
	    << "CaloHitRespoNew requires the RandomNumberGeneratorService\n"
	    "which is not present in the configuration file.  You must add the service\n"
	    "in the configuration file or remove the modules that require it.";
      }
      m_RandPoisson = new CLHEP::RandPoissonQ( rng->getEngine() );
   }
   return m_RandPoisson ;
}

CLHEP::RandGaussQ* 
CaloHitRespoNew::ranGauss() const
{
   if( 0 == m_RandGauss )
   {
      edm::Service<edm::RandomNumberGenerator> rng ;
      if ( !rng.isAvailable() ) 
      {
	 throw cms::Exception("Configuration")
	    << "CaloHitRespoNew requires the RandomNumberGeneratorService\n"
	    "which is not present in the configuration file.  You must add the service\n"
	    "in the configuration file or remove the modules that require it.";
      }
      m_RandGauss = new CLHEP::RandGaussQ( rng->getEngine() );
   }
   return m_RandGauss ;
}

const CaloSimParameters*
CaloHitRespoNew::params( const DetId& detId ) const
{
   assert( 0 != m_parameterMap ) ;
   return &m_parameterMap->simParameters( detId ) ;
}

const CaloVShape*
CaloHitRespoNew::shape() const
{
   assert( 0 != m_shape ) ;
   return m_shape ;
}

const CaloSubdetectorGeometry*
CaloHitRespoNew::geometry() const
{
   assert( 0 != m_geometry ) ;
   return m_geometry ;
}

void 
CaloHitRespoNew::setBunchRange( int minBunch , 
				int maxBunch  ) 
{
   m_minBunch = minBunch ;
   m_maxBunch = maxBunch ;
}

void 
CaloHitRespoNew::setGeometry( const CaloSubdetectorGeometry* geometry )
{
   m_geometry = geometry ;
}

void 
CaloHitRespoNew::setPhaseShift( double phaseShift )
{
   m_phaseShift = phaseShift ;
}

double
CaloHitRespoNew::phaseShift() const
{
   return m_phaseShift ;
}

void 
CaloHitRespoNew::setHitFilter( const CaloVHitFilter* filter)
{
   m_hitFilter = filter ;
}

void 
CaloHitRespoNew::setHitCorrection( const CaloVHitCorrection* hitCorrection)
{
   m_hitCorrection = hitCorrection ;
}

void 
CaloHitRespoNew::setPECorrection( const CaloVPECorrection* peCorrection )
{
   m_PECorrection = peCorrection ;
}

void 
CaloHitRespoNew::setRandomEngine( CLHEP::HepRandomEngine& engine ) const
{
   m_RandPoisson = new CLHEP::RandPoissonQ( engine ) ;
   m_RandGauss   = new CLHEP::RandGaussQ(   engine ) ;
}

const CaloSamples& 
CaloHitRespoNew::operator[]( unsigned int i ) const 
{
   assert( i < m_vSamp.size() ) ;
   return m_vSamp[ i ] ;
}

unsigned int 
CaloHitRespoNew::samplesSize() const
{
   return m_vSamp.size() ;
}

void 
CaloHitRespoNew::setupSamples( const DetId& detId )
{
   const CaloSimParameters& parameters ( *params( detId ) ) ;

   const unsigned int rSize ( parameters.readoutFrameSize() ) ;
   const unsigned int nPre  ( parameters.binOfMaximum() - 1 ) ;

   m_vSamp = VecSam( CaloGenericDetId( detId ).sizeForDenseIndexing() ) ;

   const unsigned int size ( m_vSamp.size() ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSamp[ i ].setDetId( CaloGenericDetId( detId.det(), detId.subdetId(), i ) ) ;
      m_vSamp[ i ].setSize( rSize ) ;
      m_vSamp[ i ].setPresamples( nPre ) ;
   }
}

void 
CaloHitRespoNew::blankOutUsedSamples()  // blank out previously used elements
{
   const unsigned int size ( m_index.size() ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSamp[ m_index[i] ].setBlank() ;
   }
   m_index.erase( m_index.begin() ,    // done and make ready to start over
		  m_index.end()    ) ;
}

void 
CaloHitRespoNew::run( MixCollection<PCaloHit>& hits ) 
{
   if( 0 != m_index.size() ) blankOutUsedSamples() ;

   for( MixCollection<PCaloHit>::MixItr hitItr ( hits.begin() ) ;
	hitItr != hits.end() ; ++hitItr )
   {
      if(withinBunchRange(hitItr.bunch())) {
        add(*hitItr);
      }

   }
}

void 
CaloHitRespoNew::add( const PCaloHit& hit ) 
{
      if( !edm::isNotFinite( hit.time() ) &&
	  ( 0 == m_hitFilter ||
	    m_hitFilter->accepts( hit ) ) ) putAnalogSignal( hit ) ;
}

void
CaloHitRespoNew::putAnalogSignal( const PCaloHit& hit )
{
   const DetId detId ( hit.id() ) ;

   const CaloSimParameters* parameters ( params( detId ) ) ;

   const double signal ( analogSignalAmplitude( detId, hit.energy() ) ) ;

   double time = hit.time();

   if( m_hitCorrection ) {
     time += m_hitCorrection->delay( hit ) ;
   }

   const double jitter ( time - timeOfFlight( detId ) ) ;

   const double tzero = ( shape()->timeToRise()
			  + parameters->timePhase() 
			  - jitter 
			  - BUNCHSPACE*( parameters->binOfMaximum()
					 - m_phaseShift             ) ) ;
   double binTime ( tzero ) ;

   CaloSamples& result ( *findSignal( detId ) ) ;

   const unsigned int rsize ( result.size() ) ;

   for( unsigned int bin ( 0 ) ; bin != rsize ; ++bin )
   {
      result[ bin ] += (*shape())( binTime )*signal ;
      binTime += BUNCHSPACE;
   }
}

CaloSamples* 
CaloHitRespoNew::findSignal( const DetId& detId )
{
   CaloSamples& result ( m_vSamp[ CaloGenericDetId( detId ).denseIndex() ] ) ;
   if( result.isBlank() ) m_index.push_back( &result - &m_vSamp.front() ) ;
   return &result ;
}

double 
CaloHitRespoNew::analogSignalAmplitude( const DetId& detId, float energy ) const
{
   const CaloSimParameters& parameters ( *params( detId ) ) ;

   // OK, the "energy" in the hit could be a real energy, deposited energy,
   // or pe count.  This factor converts to photoelectrons

   double npe ( energy*parameters.simHitToPhotoelectrons( detId ) ) ;

   // do we need to doPoisson statistics for the photoelectrons?
   if( parameters.doPhotostatistics() ) npe = ranPois()->fire( npe ) ;

   if( 0 != m_PECorrection ) npe = m_PECorrection->correctPE( detId, npe ) ;

   return npe ;
}

double 
CaloHitRespoNew::timeOfFlight( const DetId& detId ) const 
{
   const CaloCellGeometry* cellGeometry ( geometry()->getGeometry( detId ) ) ;
   assert( 0 != cellGeometry ) ;
   return cellGeometry->getPosition().mag()*cm/c_light ; // Units of c_light: mm/ns
}
