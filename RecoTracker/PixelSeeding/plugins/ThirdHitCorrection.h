#ifndef RecoTracker_PixelSeeding_plugins_ThirdHitCorrection_h
#define RecoTracker_PixelSeeding_plugins_ThirdHitCorrection_h

#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

class DetLayer;
class MagneticField;
class MultipleScatteringParametrisationMaker;

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

#include <optional>

class ThirdHitCorrection {
public:
  typedef PixelRecoRange<float> Range;

  ThirdHitCorrection() {}

  void init(float pt,
            const DetLayer &layer1,
            const DetLayer &layer2,
            const DetLayer &layer3,
            bool useMultipleScattering,
            const MultipleScatteringParametrisationMaker *msmaker,
            bool useBendingCorrection,
            const MagneticField *bfield);

  void init(float pt,
            const DetLayer &layer3,
            bool useMultipleScattering,
            const MultipleScatteringParametrisationMaker *msmaker,
            bool useBendingCorrection,
            const MagneticField *bfield);

  ThirdHitCorrection(float pt,
                     const DetLayer *layer,
                     const PixelRecoLineRZ &line,
                     const PixelRecoPointRZ &constraint,
                     int ol,
                     bool useMultipleScattering,
                     const MultipleScatteringParametrisationMaker *msmaker,
                     bool useBendingCorrection,
                     const MagneticField *bfield) {
    init(pt, *layer, useMultipleScattering, msmaker, useBendingCorrection, bfield);
    init(line, constraint, ol);
  }

  void init(const PixelRecoLineRZ &line, const PixelRecoPointRZ &constraint, int ol);

  void correctRPhiRange(Range &range) const {
    range.first -= theMultScattCorrRPhi;
    range.second += theMultScattCorrRPhi;
  }

  void correctRZRange(Range &range) const;

private:
  bool theBarrel;

  bool theUseMultipleScattering;
  bool theUseBendingCorrection;

  PixelRecoLineRZ theLine;
  float theMultScattCorrRPhi = 0;
  float theMScoeff = 0;
  float thePt;

  pixelrecoutilities::LongitudinalBendingCorrection theBendingCorrection;
  std::optional<MultipleScatteringParametrisation> sigmaRPhi;
};

#endif
