#ifndef EcalSimAlgos_EEShape_h
#define EcalSimAlgos_EEShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EEShape : public EcalShapeBase
{
   public:
  
      EEShape() ;

      virtual ~EEShape() ;

      virtual double threshold() const ;

   protected:
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;
};
  


#endif
  
