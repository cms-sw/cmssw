#ifndef EcalSimAlgos_APDShape_h
#define EcalSimAlgos_APDShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class APDShape : public EcalShapeBase
{
   public:
  
      APDShape( double tStart,
		double tau     ) ;

      virtual ~APDShape() ;

      virtual double threshold() const ;

   protected:
  
      virtual void fillShape( EcalShapeBase::DVec& aVec ) const ;

   private:

      double m_tStart ;
      double m_tau    ;
};
  


#endif
  
