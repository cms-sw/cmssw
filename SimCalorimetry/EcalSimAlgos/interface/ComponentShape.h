#ifndef EcalSimAlgos_ComponentShape_h
#define EcalSimAlgos_ComponentShape_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "CondFormats/EcalObjects/interface/EcalSimComponentShape.h"
#include "CondFormats/DataRecord/interface/EcalSimComponentShapeRcd.h"

class ComponentShape : public EcalShapeBase {
public:
  // useDB = false
  ComponentShape(int shapeIndex) : EcalShapeBase(false), shapeIndex_(shapeIndex) { buildMe(nullptr, false); }
  // useDB = true, buildMe is executed when setEventSetup and DB conditions are available
  ComponentShape(int shapeIndex, edm::ESGetToken<EcalSimComponentShape, EcalSimComponentShapeRcd> espsToken)
      : EcalShapeBase(true), espsToken_(espsToken), shapeIndex_(shapeIndex) {}

  // override EcalShapeBase timeToRise, so that it does not align component shapes to same peaking time
  double timeToRise() const override;

protected:
  void fillShape(float& time_interval,
                 double& m_thresh,
                 EcalShapeBase::DVec& aVec,
                 const edm::EventSetup* es) const override;

private:
  edm::ESGetToken<EcalSimComponentShape, EcalSimComponentShapeRcd> espsToken_;
  int shapeIndex_;
};

#endif
