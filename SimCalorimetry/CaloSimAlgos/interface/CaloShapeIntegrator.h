#ifndef CaloSimAlgos_CaloShapeIntegrator_h
#define CaloSimAlgos_CaloShapeIntegrator_h

/**  This class takes an existing Shape, and
     integrates it, summing up all the values,
     each nanosecond, up to the bunch spacing
*/

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"


class CaloShapeIntegrator: public CaloVShape
{
   public:

      enum {BUNCHSPACE = 25};

      CaloShapeIntegrator( const CaloVShape* aShape ) ;

      virtual ~CaloShapeIntegrator() ;

      virtual double operator () ( double startTime ) const ;
      virtual double timeToRise()                     const ;

   private:

      const CaloVShape* m_shape ;
};

#endif

