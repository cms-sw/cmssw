#include <cmath>

#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"

#include<assert.h>

APDShape::~APDShape()
{
}

APDShape::APDShape( double tStart,
		    double tau     ) :
   EcalShapeBase( true ) ,
   m_tStart ( tStart ) ,
   m_tau    ( tau    )
{
   assert( m_tau    > 1.e-5 ) ;
   assert( m_tStart > 0     ) ;
   buildMe() ;
}

double
APDShape::threshold() const
{
   return 0.0 ; 
}

void
APDShape::fillShape( EcalShapeBase::DVec& aVec ) const
{
   for( unsigned int i ( 0 ) ; i != k1NSecBinsTotal ; ++i )
   {
      const double ctime ( ( 1.*i + 0.5 - m_tStart )/m_tau ) ;
      aVec[i] = ( 0 > ctime ? 0 : ctime * exp( 1. - ctime ) ) ;
   }
}
