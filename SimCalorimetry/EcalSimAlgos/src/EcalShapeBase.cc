#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

const double EcalShapeBase::qNSecPerBin = 1./(1.*kNBinsPerNSec) ;

EcalShapeBase::~EcalShapeBase()
{
   delete m_derivPtr ;
}

EcalShapeBase::EcalShapeBase( bool   aSaveDerivative ) :
   m_firstIndexOverThreshold ( 0   ) ,
   m_firstTimeOverThreshold  ( 0.0 ) ,
   m_indexOfMax              ( 0   ) ,
   m_timeOfMax               ( 0.0 ) ,
   m_shape                   ( DVec( kNBinsStored, 0.0 ) ) ,
   m_derivPtr                ( aSaveDerivative ? new DVec( kNBinsStored, 0.0 ) : 0 )
{
}

double 
EcalShapeBase::timeOfThr()  const 
{
   return m_firstTimeOverThreshold ;
 }

double 
EcalShapeBase::timeOfMax()  const 
{
   return m_timeOfMax              ; 
}

double 
EcalShapeBase::timeToRise() const 
{
   return timeOfMax() - timeOfThr() ;
}


void
EcalShapeBase::buildMe()
{
   DVec shapeArray( k1NSecBinsTotal , 0.0 ) ;

   fillShape( shapeArray ) ;

   const double maxel ( *max_element( shapeArray.begin(), shapeArray.end() ) ) ;

   const double maxelt ( 1.e-5 < maxel ? maxel : 1 ) ;

   for( unsigned int i ( 0 ) ; i != shapeArray.size(); ++i )
   {
      shapeArray[i] = shapeArray[i]/maxelt ;
   } 

   const double thresh ( threshold()/maxelt ) ;

/*
   for( unsigned int i ( 0 ) ; i != k1NSecBinsTotal ; ++i ) 
   {
      LogDebug("EcalShapeBase") << " time (ns) = " << (double)i << " tabulated ECAL pulse shape = " << shapeArray[i];
   }
*/

   const double delta ( qNSecPerBin/2. ) ;

   for( unsigned int j ( 0 ) ; j != kNBinsStored ; ++j )
   {
      const double xb ( ( j + 0.5 )*qNSecPerBin ) ; 

      const unsigned int ibin ( j/kNBinsPerNSec ) ;

      double value = 0.0 ;
      double deriv = 0.0 ;

      if( 0                 ==     ibin ||
	  shapeArray.size() == 1 + ibin    ) // cannot do quadratic interpolation at ends
      {
	 value = shapeArray[ibin]; 
	 deriv = 0.0 ;
      }
      else 
      {
	 const double x  ( xb - ( ibin + 0.5 ) ) ;
	 const double f1 ( shapeArray[ ibin - 1 ] ) ;
	 const double f2 ( shapeArray[ ibin     ] ) ;
	 const double f3 ( shapeArray[ ibin + 1 ] ) ;
	 const double a  ( f2 ) ;
	 const double b  ( ( f3 - f1 )/2. ) ;
	 const double c  ( ( f1 + f3 )/2. - f2 ) ;
	 value = a + b*x + c*x*x;
	 deriv = ( b + 2*c*x )/delta ;
      }

      m_shape[ j ] = value;
      if( 0 != m_derivPtr ) (*m_derivPtr)[ j ] = deriv;

      if( 0      <  j                         &&
	  thresh <  value                     &&
	  0      == m_firstIndexOverThreshold     )
      {
	 m_firstIndexOverThreshold = j - 1 ;
	 m_firstTimeOverThreshold  = m_firstIndexOverThreshold*qNSecPerBin ;
      }

      if( m_shape[ m_indexOfMax ] < value )
      {
	 m_indexOfMax = j ;
      }

//      LogDebug("EcalShapeBase") << " time (ns) = " << ( j + 1.0 )*qNSecPerBin - delta 
//				<< " interpolated ECAL pulse shape = " << m_shape[ j ] 
//				<< " derivative = " << ( 0 != m_derivPtr ? (*m_derivPtr)[ j ] : 0 ) ;
   }
   m_timeOfMax = m_indexOfMax*qNSecPerBin ;
}

unsigned int
EcalShapeBase::timeIndex( double aTime ) const
{
   const int index ( m_firstIndexOverThreshold +
		     (unsigned int) ( aTime*kNBinsPerNSec + 0.5 ) ) ;

   const bool bad ( (int) m_firstIndexOverThreshold >  index || 
		    (int) kNBinsStored              <= index    ) ;

   if(		    (int) kNBinsStored              <= index    )
   {
      LogDebug("EcalShapeBase") << " ECAL MGPA shape requested for out of range time " << aTime ;
   }
   return ( bad ? kNBinsStored : (unsigned int) index ) ;
}

double 
EcalShapeBase::operator() ( double aTime ) const
{
   // return pulse amplitude for request time in ns

   const unsigned int index ( timeIndex( aTime ) ) ;
   return ( kNBinsStored == index ? 0 : m_shape[ index ] ) ;
}

double 
EcalShapeBase::derivative( double aTime ) const
{
   const unsigned int index ( timeIndex( aTime ) ) ;
   return ( 0            == m_derivPtr ||
	    kNBinsStored == index         ? 0 : (*m_derivPtr)[ index ] ) ;
}
