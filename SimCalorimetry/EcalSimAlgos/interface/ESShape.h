#ifndef EcalSimAlgos_ESShape_h
#define EcalSimAlgos_ESShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

/* \class ESShape
 * \brief preshower pulse-shape
 * 
 * Preshower pulse shape
 * - Gain = 1 : shape for low gain for data taking
 * - Gain = 2 : shape for high gain for calibration and low energy runs
 * 
 * Preshower three time samples happen at -5, 20 and 45 ns 
 *
 */
class ESShape : public CaloVShape {
public:
  /// ctor
  ESShape();
  /// dtor
  ~ESShape() override {}

  void setGain(const int gain) { theGain_ = gain; }
  double operator()(double time) const override;
  double timeToRise() const override;

  void display() const {}

private:
  int theGain_;
};

#endif
