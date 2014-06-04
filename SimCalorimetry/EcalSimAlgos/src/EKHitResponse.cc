#include "SimCalorimetry/EcalSimAlgos/interface/EKHitResponse.h" 
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"


EKHitResponse::EKHitResponse( const CaloVSimParameterMap* parameterMap , 
			      const CaloVShape*           shape         ) :

   EcalHitResponse( parameterMap, shape )
{
   const EKDetId detId ( EKDetId::detIdFromDenseIndex( 0 ) ) ;
   const CaloSimParameters& parameters ( parameterMap->simParameters( detId ) ) ;

   const unsigned int rSize ( parameters.readoutFrameSize() ) ;
   const unsigned int nPre  ( parameters.binOfMaximum() - 1 ) ;

   const unsigned int size ( EKDetId::kSizeForDenseIndexing ) ;

   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSam.emplace_back(CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
		    rSize, nPre ) ;
   }
}

EKHitResponse::~EKHitResponse()
{
}

unsigned int
EKHitResponse::samplesSize() const
{
   return m_vSam.size() ;
}

unsigned int
EKHitResponse::samplesSizeAll() const
{
   return m_vSam.size() ;
}

const EcalHitResponse::EcalSamples* 
EKHitResponse::operator[]( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EKHitResponse::operator[]( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EKHitResponse::vSam( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalHitResponse::EcalSamples* 
EKHitResponse::vSamAll( unsigned int i )
{
   return &m_vSam[ i ] ;
}

const EcalHitResponse::EcalSamples* 
EKHitResponse::vSamAll( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}
