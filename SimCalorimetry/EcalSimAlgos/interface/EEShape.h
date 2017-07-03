#ifndef EcalSimAlgos_EEShape_h
#define EcalSimAlgos_EEShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EEShape : public EcalShapeBase
{
   public:
  
      EEShape() ;

      ~EEShape() override ;

      double threshold() const override ;

   protected:
  
      void fillShape( EcalShapeBase::DVec& aVec ) const override ;
};
  


#endif
  
