#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"

CaloShapeIntegrator::CaloShapeIntegrator( const CaloVShape* aShape ) :
   m_shape ( aShape )
{
}

CaloShapeIntegrator::~CaloShapeIntegrator() 
{
}

double 
CaloShapeIntegrator::timeToRise() const 
{
   return m_shape->timeToRise() ;
}

double 
CaloShapeIntegrator::operator() ( double startTime ) const 
{
   double sum = 0.;

   double time = startTime + 0.5 ;

   for( unsigned istep = 0; istep < BUNCHSPACE ; ++istep )
   {
      sum += (*m_shape)( time ) ;
      time = time + 1.0 ;
   }
   return sum;
}


