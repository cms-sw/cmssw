#ifndef EcalSimAlgos_EBShape_h
#define EcalSimAlgos_EBShape_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

class EBShape : public EcalShapeBase {
public:
  // useDB = false
  EBShape() : EcalShapeBase(false) { buildMe(); }
  // useDB = true, buildMe is executed when setEventSetup and DB conditions are available
  EBShape(edm::ConsumesCollector iC) : EcalShapeBase(true), espsToken_(iC.esConsumes()) {}

protected:
  void fillShape(float& time_interval,
                 double& m_thresh,
                 EcalShapeBase::DVec& aVec,
                 const edm::EventSetup* es) const override;

private:
  edm::ESGetToken<EcalSimPulseShape, EcalSimPulseShapeRcd> espsToken_;
};

#endif
