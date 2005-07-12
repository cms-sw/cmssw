#ifndef SimG4Core_G4SimVertex_H
#define SimG4Core_G4SimVertex_H

#include "CLHEP/Vector/ThreeVector.h"

class G4SimVertex
{
public:
    G4SimVertex() {}
    G4SimVertex(const Hep3Vector & ip, double it, int iv) : 
	ilv_(ip),itime_(it),itrack_(iv) {}
    /// index of the parent (-1 if no parent)
    const Hep3Vector & vertexPosition() const { return  ilv_; }
    const double vertexGlobalTime() const     { return  itime_; }
    const int parentIndex() const	      { return  itrack_; }
private:
    Hep3Vector ilv_;
    double itime_;
    int itrack_;
};

#endif
