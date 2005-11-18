#ifndef CaloVHitCorrection_h
#define CaloVHitCorrection_h

namespace cms {

class CaloVHitCorrection {
public:
  virtual void correct(const PCaloHit & hit) const = 0;
};

}
#endif

