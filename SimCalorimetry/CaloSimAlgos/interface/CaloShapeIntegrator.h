#ifndef CaloSimAlgos_CaloShapeIntegrator_h
#define CaloSimAlgos_CaloShapeIntegrator_h

/**  This class takes an existing Shape, and
     integrates it, summing up all the values,
     each nanosecond, up to the bunch spacing
*/

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

class CaloShapeIntegrator : public CaloVShape {
public:
  enum { BUNCHSPACE = 25 };

  CaloShapeIntegrator(const CaloVShape *aShape);

  ~CaloShapeIntegrator() override;

  double operator()(double startTime) const override;
  double timeToRise() const override;

private:
  const CaloVShape *m_shape;
};

#endif
