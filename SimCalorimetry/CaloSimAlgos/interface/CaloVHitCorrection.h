#ifndef CaloSimAlgos_CaloVHitCorrection_h
#define CaloSimAlgos_CaloVHitCorrection_h

class CaloVHitCorrection {
public:
  virtual void correct(const PCaloHit & hit) const = 0;
};

#endif

