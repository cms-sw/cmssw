#ifndef EcalSimAlgos_EEShape_h
#define EcalSimAlgos_EEShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EEShape : public EcalShapeBase
{
   public:
  
      EEShape( bool   aSaveDerivative = false ) ;

      virtual ~EEShape() ;
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;

      virtual double threshold() const ;
};
  


#endif
  
