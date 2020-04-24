#ifndef EcalSimAlgos_EBShape_h
#define EcalSimAlgos_EBShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EBShape : public EcalShapeBase
{
   public:
  
      EBShape() ;

      ~EBShape() override ;

      double threshold() const override ;

   protected:
  
      void fillShape( EcalShapeBase::DVec& aVec ) const override ;
};
  


#endif
  
