#ifndef SimG4Core_G4SimVertex_H
#define SimG4Core_G4SimVertex_H

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector> 
#include <cmath>

class G4SimVertex
{
public:
    G4SimVertex() {}
    G4SimVertex(const math::XYZVectorD & ip, double it, int iv, unsigned int typ=0) : 
      ilv_(ip),itime_(it),itrack_(iv),procType_(typ) {}
    /// index of the parent (-1 if no parent)
    const math::XYZVectorD & vertexPosition() const { return  ilv_; }
    double vertexGlobalTime() const     { return  itime_; }
    int parentIndex() const	      { return  itrack_; }
    unsigned int processType() const    { return  procType_; }
private:
    math::XYZVectorD ilv_;
    double itime_;
    int itrack_;
    unsigned int procType_;
};

#endif
