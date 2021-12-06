#ifndef CastorSim_CastorShape_h
#define CastorSim_CastorShape_h
#include <vector>

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

/**

   \class CastorShape

   \brief  shaper for Castor

*/

class CastorShape : public CaloVShape {
public:
  CastorShape();
  CastorShape(const CastorShape &d);

  ~CastorShape() override {}

  double operator()(double time) const override;
  double timeToRise() const override;

private:
  void computeShapeCastor();

  int nbin_;
  std::vector<float> nt_;
};

#endif
