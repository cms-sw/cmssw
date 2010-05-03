#ifndef EcalSimAlgos_ESShape_h
#define EcalSimAlgos_ESShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

/* \class ESShape
 * \brief preshower pulse-shape
 * 
 * Preshower pulse shape
 * - Gain = 0 : old shape used in ORCA
 * - Gain = 1 : shape for low gain for data taking
 * - Gain = 2 : shape for high gain for calibration
 * 
 * Preshower three time samples happen at -5, 20 and 45 ns 
 *
 */                                                                                            
class ESShape : public CaloVShape 
{
  
 public:
  
  /// ctor
  ESShape(int Gain);
  /// dtor
  ~ESShape(){}
  
      virtual double operator () (double time) const;
      virtual double timeToRise()              const ;

  void display () const {}

 private:

  int theGain_;
  double A_;
  double Qcf_;
  double omegac_;
  double norm_;
  double M_;

};

#endif
