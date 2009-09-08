#ifndef EcalSimAlgos_EBShape_h
#define EcalSimAlgos_EBShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EBShape : public EcalShapeBase
{
   public:
  
      EBShape( double aTimePhase ,
	       bool   aSaveDerivative = false ) ;

      virtual ~EBShape() ;
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;

      virtual double threshold() const ;
};
  


#endif
  
