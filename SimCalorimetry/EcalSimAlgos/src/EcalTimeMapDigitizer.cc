#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVPECorrection.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 
#include <iostream>



EcalTimeMapDigitizer::EcalTimeMapDigitizer(EcalSubdetector myDet):
  m_subDet(myDet),
  m_geometry(0)
{
//    edm::Service<edm::RandomNumberGenerator> rng ;
//    if ( !rng.isAvailable() ) 
//    {
//       throw cms::Exception("Configuration")
// 	 << "EcalTimeMapDigitizer requires the RandomNumberGeneratorService\n"
// 	 "which is not present in the configuration file.  You must add the service\n"
// 	 "in the configuration file or remove the modules that require it.";
//    }
//    m_RandPoisson = new CLHEP::RandPoissonQ( rng->getEngine() ) ;
//    m_RandGauss   = new CLHEP::RandGaussQ(   rng->getEngine() ) ;

  unsigned int size=0;
  DetId detId(0);

  //Initialize the map
  if (myDet==EcalBarrel)
    {
      size=EBDetId::kSizeForDenseIndexing;
      detId=EBDetId::detIdFromDenseIndex( 0 ) ;
    }
  else if (myDet==EcalEndcap)
    {
      size=EEDetId::kSizeForDenseIndexing;
      detId=EEDetId::detIdFromDenseIndex( 0 ) ;
    }
  else 
    std::cout << "[EcalTimeMapDigitizer]::ERROR::This subdetector " << myDet << " is not implemented" <<  std::endl;


   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSam.emplace_back(CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
			  m_maxBunch-m_minBunch+1, abs(m_minBunch) );
   }
}

EcalTimeMapDigitizer::~EcalTimeMapDigitizer()
{
}

void
EcalTimeMapDigitizer::add(const std::vector<PCaloHit> & hits, int bunchCrossing) {
  if(bunchCrossing>=m_minBunch && bunchCrossing<=m_maxBunch ) {

    for(std::vector<PCaloHit>::const_iterator it = hits.begin(), itEnd = hits.end(); it != itEnd; ++it) {
      //here goes the map logic
      if (edm::isNotFinite( (*it).time() ) )
	continue;

      //Just consider only the hits belonging to the specified time layer
      if ((*it).depth()!=m_timeLayerId)
	continue;
      
//       const DetId detId ( (*it).id() ) ;
      
//       double time = (*it).time();
      
//       const double jitter ( time - timeOfFlight( detId, m_timeLayerId ) ) ;

//       TimeSamples& result ( *findSignal( detId ) ) ;

//       const unsigned int rsize ( result.size() ) ;

      //here fill the result for the given bunch crossing

    }
  }
}


EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::findSignal( const DetId& detId )
{
   const unsigned int di ( CaloGenericDetId( detId ).denseIndex() ) ;
   TimeSamples* result ( vSamAll( di ) ) ;
   if( result->zero() ) m_index.push_back( di ) ;
   return result ;
}



void 
EcalTimeMapDigitizer::setGeometry( const CaloSubdetectorGeometry* geometry )
{
   m_geometry = geometry ;
}


void 
EcalTimeMapDigitizer::blankOutUsedSamples()  // blank out previously used elements
{
   const unsigned int size ( m_index.size() ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      vSamAll( m_index[i] )->setZero() ;
   }

   m_index.erase( m_index.begin() ,    // done and make ready to start over
		  m_index.end()    ) ;
}


void
EcalTimeMapDigitizer::finalizeHits()
{
}

void
EcalTimeMapDigitizer::initializeMap()
{
   blankOutUsedSamples();
}


void 
EcalTimeMapDigitizer::run( EcalTimeDigiCollection& output  )
{
}


double 
EcalTimeMapDigitizer::timeOfFlight( const DetId& detId , int layer) const 
{
  //not using the layer yet
   const CaloCellGeometry* cellGeometry ( m_geometry->getGeometry( detId ) ) ;
   assert( 0 != cellGeometry ) ;
   return cellGeometry->getPosition().mag()*cm/c_light ; // Units of c_light: mm/ns
}


unsigned int
EcalTimeMapDigitizer::samplesSize() const
{
   return m_vSam.size() ;
}

unsigned int
EcalTimeMapDigitizer::samplesSizeAll() const
{
   return m_vSam.size() ;
}

const EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::operator[]( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::operator[]( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSam( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSamAll( unsigned int i )
{
   return &m_vSam[ i ] ;
}

const EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSamAll( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

int 
EcalTimeMapDigitizer::minBunch() const 
{
  return m_minBunch ; 
}

int 
EcalTimeMapDigitizer::maxBunch() const 
{
  return m_maxBunch ; 
}

EcalTimeMapDigitizer::VecInd& 
EcalTimeMapDigitizer::index() 
{
  return m_index ; 
}

const EcalTimeMapDigitizer::VecInd& 
EcalTimeMapDigitizer::index() const
{
  return m_index ; 
}

// const EcalTimeMapDigitizer::TimeSamples* 
// EcalTimeMapDigitizer::findDetId( const DetId& detId ) const
// {
//    const unsigned int di ( CaloGenericDetId( detId ).denseIndex() ) ;
//    return vSamAll( di ) ;
// }
