#include "SimCalorimetry/EcalSimAlgos/interface/ESHitResponse.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"


ESHitResponse::ESHitResponse( const CaloVSimParameterMap* parameterMap , 
			      const CaloVShape*           shape          ) :
   EcalHitResponse( parameterMap , shape )
{
   assert( 0 != parameterMap ) ;
   assert( 0 != shape ) ;
   const ESDetId detId ( ESDetId::detIdFromDenseIndex( 0 ) ) ;
   const CaloSimParameters& parameters ( parameterMap->simParameters( detId ) ) ;

   const unsigned int rSize ( parameters.readoutFrameSize() ) ;
   const unsigned int nPre  ( parameters.binOfMaximum() - 1 ) ;

   const unsigned int size ( ESDetId::kSizeForDenseIndexing ) ;

   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSam.emplace_back(
	 CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
		    rSize, nPre ) ;
   }
}

ESHitResponse::~ESHitResponse()
{
}

unsigned int
ESHitResponse::samplesSize() const
{
   return index().size() ;
}

unsigned int
ESHitResponse::samplesSizeAll() const
{
   return ESDetId::kSizeForDenseIndexing ;
}

const EcalHitResponse::EcalSamples* 
ESHitResponse::operator[]( unsigned int i ) const
{
   return &m_vSam[ index()[ i ] ] ;
}

EcalHitResponse::EcalSamples* 
ESHitResponse::operator[]( unsigned int i )
{
   return &m_vSam[ index()[ i ] ] ;
}

EcalHitResponse::EcalSamples* 
ESHitResponse::vSam( unsigned int i )
{
   return &m_vSam[ index()[ i ] ] ;
}

EcalHitResponse::EcalSamples* 
ESHitResponse::vSamAll( unsigned int i )
{
   return &m_vSam[ i ] ;
}

const EcalHitResponse::EcalSamples* 
ESHitResponse::vSamAll( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}
