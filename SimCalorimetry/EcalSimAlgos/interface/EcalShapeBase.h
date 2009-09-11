#ifndef EcalSimAlgos_EcalShapeBase_h
#define EcalSimAlgos_EcalShapeBase_h

#include<vector>
//#include<stdexcept>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**
   \class EcalShape
   \brief  shaper for Ecal
*/
class EcalShapeBase : public CaloVShape
{
   public:

      typedef std::vector<double> DVec ;
  
      EcalShapeBase( bool   aSaveDerivative ) ;

      virtual ~EcalShapeBase() ;

      double operator() ( double aTime ) const ;

      double timeOfThr()  const { return m_firstTimeOverThreshold ; }
      double timeOfMax()  const { return m_timeOfMax              ; }
      double timeToRise() const { return timeOfMax() - timeOfThr(); }

      virtual void   fillShape( DVec& aVec ) const = 0 ;

      virtual double threshold()             const = 0 ;
  
      double derivative ( double time ) const ; // appears to not be used anywhere

      enum { kReadoutTimeInterval = 25 , // in nsec
	     kNBinsPerNSec        = 10 , // granularity of internal array
	     k1NSecBins           = kReadoutTimeInterval*kNBinsPerNSec ,
	     k1NSecBinsTotal      = 2*k1NSecBins ,
	     kNBinsStored         = k1NSecBinsTotal*kNBinsPerNSec
      } ;

      static const double qNSecPerBin ;

   protected:

      void buildMe() ;

      unsigned int timeIndex( double aTime ) const ;

   private:

      unsigned int m_firstIndexOverThreshold ;
      double       m_firstTimeOverThreshold  ;
      unsigned int m_indexOfMax ;
      double       m_timeOfMax  ;
      DVec  m_shape ;
      DVec* m_derivPtr ;
};
  


#endif
  
