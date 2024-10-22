///////////////////////////////////////////////////////////////////////////////
// File: FP420NumberingScheme.h
// Date: 02.2006
// Description: Numbering scheme for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef FP420NumberingScheme_h
#define FP420NumberingScheme_h

#include <map>

class G4Step;
class G4String;

class FP420NumberingScheme {
public:
  FP420NumberingScheme();
  ~FP420NumberingScheme() = default;

  unsigned int getUnitID(const G4Step* aStep) const;

  static unsigned int packFP420Index(int det, int zside, int station, int superplane);
  static void unpackFP420Index(const unsigned int& idx, int& det, int& zside, int& station, int& superplane);

  static unsigned packMYIndex(int rn0, int pn0, int sn0, int det, int zside, int sector, int zmodule) {
    int zScale = (rn0 - 1);           // rn0=3 - current --> for update  rn0=7
    int sScale = zScale * (pn0 - 1);  //pn0=6
    int dScale = sScale * (sn0 - 1);  //sn0=3
    unsigned int intindex = dScale * (det - 1) + sScale * (sector - 1) + zScale * (zmodule - 1) + zside;
    return intindex;
  }

  static void unpackMYIndex(const int& idx, int rn0, int pn0, int sn0, int& det, int& zside, int& sector, int& zmodule) {
    int zScale = (rn0 - 1), sScale = (rn0 - 1) * (pn0 - 1), dScale = (rn0 - 1) * (pn0 - 1) * (sn0 - 1);
    det = (idx - 1) / dScale + 1;
    sector = (idx - 1 - dScale * (det - 1)) / sScale + 1;
    zmodule = (idx - 1 - dScale * (det - 1) - sScale * (sector - 1)) / zScale + 1;
    zside = idx - dScale * (det - 1) - sScale * (sector - 1) - zScale * (zmodule - 1);
  }

  static int unpackLayerIndex(int rn0, int zside) {
    // 1,2
    int layerIndex = 1, b;
    float a = (zside + 1) / 2.;
    b = (int)a;
    if (a - b != 0.)
      layerIndex = 2;
    //
    if (zside > (rn0 - 1) || zside < 1)
      layerIndex = 0;
    return layerIndex;
  }

  static int unpackCopyIndex(int rn0, int zside) {
    // 1,2,3
    int copyIndex = 0;
    if (zside <= (rn0 - 1) && zside >= 1) {
      int layerIndex = 1;
      float a = (zside + 1) / 2.;
      int b = (int)a;
      if (a - b != 0.)
        layerIndex = 2;
      if (layerIndex == 2)
        copyIndex = zside / 2;
      if (layerIndex == 1)
        copyIndex = (zside + 1) / 2;
    }
    return copyIndex;
  }

  static int unpackOrientation(int rn0, int zside) {
    // Front: Orientation= 1; Back: Orientation= 2
    int Orientation = 2;
    if (zside > (rn0 - 1) || zside < 1)
      Orientation = 0;
    if (zside == 1 || zside == 2)
      Orientation = 1;
    //
    return Orientation;
  }

  static int realzside(int rn0, int zsideinorder) {
    // zsideinorder:1 2 3 4 5 6
    //sensorsold    1 0 0 2 0 0
    //sensorsnew    1 0 5 2 4 0 ???
    //sensorsnew    1 3 0 2 0 6
    //zside         1 3 5 2 4 6  over layers 1 and 2
    int zside, zsidereal;
    if (zsideinorder < 0) {
      zside = 0;
    } else if (zsideinorder < 4) {
      zside = 2 * zsideinorder - 1;
    } else if (zsideinorder < 7) {
      zside = 2 * zsideinorder - 6;
    } else {
      zside = 0;
    }
    zsidereal = zside;
    //
    //old:
    if (rn0 == 3) {
      if (zside > 2)
        zsidereal = 0;
    }
    //new:
    if (rn0 == 7) {
      if (zside == 4 || zside == 5)
        zsidereal = 0;
    }
    return zsidereal;
  }
};

#endif
