#include "SimCalorimetry/EcalSimAlgos/interface/EEHitResponse.h" 
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
   const ESDetId detId ( ESDetId::detIdFromDenseIndex( 0 ) ) ;
   const CaloSimParameters& parameters ( *parameterMap( detId ) ) ;

   const unsigned int rSize ( parameters.readoutFrameSize() ) ;
   const unsigned int nPre  ( parameters.binOfMaximum() - 1 ) ;

   const unsigned int size ( ESDetId::kSizeForDenseIndexing ) ;

   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
      m_vSam.push_back(
	 ESSamples( CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
		    rSize, nPre ) ) ;
   }
}

ESHitResponse::~ESHitResponse()
{
}

unsigned int
EBHitResponse::samplesSize() const
{
   return index().size() ;
}

unsigned int
EBHitResponse::samplesSizeAll() const
{
   return m_vSam().size() ;
}

const EcalHitResponse::EcalSamples* 
ESHitResponse::operator[]( unsigned int i ) const
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
