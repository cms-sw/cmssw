#ifndef EcalSimAlgos_APDShape_h
#define EcalSimAlgos_APDShape_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class APDShape : public EcalShapeBase {
public:
  APDShape(bool useDB) : EcalShapeBase(useDB) {
    if (!useDB)
      buildMe();
  }  // if useDB = true, then buildMe is executed when setEventSetup and DB conditions are available
     //   APDShape():EcalShapeBase(false){;}

protected:
  void fillShape(float& time_interval,
                 double& m_thresh,
                 EcalShapeBase::DVec& aVec,
                 const edm::EventSetup* es) const override;
};

#endif
