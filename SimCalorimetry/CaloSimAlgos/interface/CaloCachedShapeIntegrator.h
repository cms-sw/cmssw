#ifndef CaloSimAlgos_CaloCachedShapeIntegrator_h
#define CaloSimAlgos_CaloCachedShapeIntegrator_h

/**  This class takes an existing Shape, and
     integrates it, summing up all the values,
     each nanosecond, up to the bunch spacing
*/

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <vector>

class CaloCachedShapeIntegrator: public CaloVShape
{
   public:

      CaloCachedShapeIntegrator( const CaloVShape* aShape ) ;

      virtual ~CaloCachedShapeIntegrator() ;

      virtual double operator () ( double startTime ) const ;
      virtual double timeToRise()                     const ;

   private:

      std::vector<double> v_;
      double timeToRise_;
};

#endif

