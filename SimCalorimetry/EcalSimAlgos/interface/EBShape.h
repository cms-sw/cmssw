#ifndef EcalSimAlgos_EBShape_h
#define EcalSimAlgos_EBShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EBShape : public EcalShapeBase {
public:
  EBShape(bool useDB) : EcalShapeBase(useDB) {
    if (!useDB)
      buildMe();
  }  // if useDB = true, then buildMe is executed when setEventSetup and DB conditions are available
  //EBShape():EcalShapeBase(false){;}

protected:
  void fillShape(float& time_interval,
                 double& m_thresh,
                 EcalShapeBase::DVec& aVec,
                 const edm::EventSetup* es) const override;
};

#endif
