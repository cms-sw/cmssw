#ifndef TkHitPairsCellManager_H
#define TkHitPairsCellManager_H
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"

class TkHitPairsCellManager {
 public: 
  TkHitPairsCellManager(int sizeX, int sizeY) 
    : theNX(sizeX), theNY(sizeY)
  { theCells = new TkHitPairsCacheCell* [theNX]; 
    for (int i=0; i<theNX; i++) theCells[i] = new TkHitPairsCacheCell[theNY]; }
  
  ~TkHitPairsCellManager() 
   { for (int i=0; i<theNX; i++) delete [] theCells[i]; delete [] theCells; }

  TkHitPairsCacheCell & operator () (int ix, int iy)
   { return theCells[ix][iy]; }
private:
  int theNX, theNY;
  TkHitPairsCacheCell ** theCells;
};
#endif
