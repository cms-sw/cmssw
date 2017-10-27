#ifndef EcalSimAlgos_APDShape_h
#define EcalSimAlgos_APDShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class APDShape : public EcalShapeBase
{
   public:
  
      APDShape( double tStart,
		double tau     ) ;

      ~APDShape() override ;

      double threshold() const override ;

   protected:
  
      void fillShape( EcalShapeBase::DVec& aVec ) const override ;

   private:

      double m_tStart ;
      double m_tau    ;
};
  


#endif
  
