#ifndef EventShapeVariables_h
#define EventShapeVariables_h

#include <memory>
#include <string>
#include <iostream>

#include "TMath.h"
#include "TVector3.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"

class EventShapeVariables {

 public:
  
  explicit EventShapeVariables();
  ~EventShapeVariables(){};

  int nStep() {return nStep_;};  
  double sphericity()  const {return sph_; };
  double aplanarity()  const {return apl_; };
  double circularity() const {return cir_; };
  double isotropy()    const {return iso_; };

  void setNStep(int value) {nStep_=value;};
  double sphericity(const std::vector<TVector3>&);
  double aplanarity(const std::vector<TVector3>&);
  double circularity(const std::vector<TVector3>&);
  double isotropy(const std::vector<TVector3>&);
  
 private:

  TMatrixDSym momentumTensor(const std::vector<TVector3>&);

 private:

  int nStep_;
  double sph_;
  double apl_;
  double cir_;
  double iso_;
};

#endif

