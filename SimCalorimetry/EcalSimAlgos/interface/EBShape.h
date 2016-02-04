#ifndef EcalSimAlgos_EBShape_h
#define EcalSimAlgos_EBShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EBShape : public EcalShapeBase
{
   public:
  
      EBShape() ;

      virtual ~EBShape() ;

      virtual double threshold() const ;

   protected:
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;
};
  


#endif
  
