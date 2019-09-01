#ifndef InduceChargeFP420_h
#define InduceChargeFP420_h

#include "SimRomanPot/SimFP420/interface/IChargeFP420.h"
//
//
class InduceChargeFP420 : public IChargeFP420 {
  //
  //
public:
  InduceChargeFP420(double w, double g) {
    clusterWidth = w;
    geVperElectron = g;
  }
  ~InduceChargeFP420() override {}
  //
  //
  IChargeFP420::hit_map_type induce(const CDrifterFP420::collection_type &,
                                    int numStrips,
                                    double localPitch,
                                    int numStripsW,
                                    double localPitchW,
                                    int xytype,
                                    int verbosity) override;
  //
  //
private:
  std::vector<float> signalCoupling;

  double clusterWidth;
  double geVperElectron;
};

#endif
