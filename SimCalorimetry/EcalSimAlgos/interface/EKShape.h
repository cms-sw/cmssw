#ifndef EcalSimAlgos_EKShape_h
#define EcalSimAlgos_EKShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EKShape : public EcalShapeBase
{
   public:
  
      EKShape() ;

      virtual ~EKShape() ;

      virtual double threshold() const ;

   protected:
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;
};
  


#endif
  
