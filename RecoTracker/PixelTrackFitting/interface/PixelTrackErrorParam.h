#ifndef RecoTracker_PixelTrackFitting_PixelTrackErrorParam_h
#define RecoTracker_PixelTrackFitting_PixelTrackErrorParam_h
#include <cmath>

class PixelTrackErrorParam {
public:
  PixelTrackErrorParam(double eta, double pt);
  double errPt() const;
  // { return ptPar0_[theIEta]*thePt + ptPar1_[theIEta]*thePt2; }
  inline double errCot() const { return cotPar0_[theIEta] + cotPar1_[theIEta] / thePt + cotPar2_[theIEta] / thePt2; }
  inline double errTip() const { return sqrt(tipPar0_[theIEta] + tipPar1_[theIEta] / thePt2); }
  inline double errZip() const { return sqrt(zipPar0_[theIEta] + zipPar1_[theIEta] / thePt2); }
  inline double errPhi() const { return sqrt(phiPar0_[theIEta] + phiPar1_[theIEta] / thePt2); }

private:
  unsigned int theIEta;
  double thePt, thePt2;

  static const unsigned int nEta = 25;
  static double const ptPar0_[nEta];
  static double const ptPar1_[nEta];
  static double const cotPar0_[nEta];
  static double const cotPar1_[nEta];
  static double const cotPar2_[nEta];
  static double const tipPar0_[nEta];
  static double const tipPar1_[nEta];
  static double const zipPar0_[nEta];
  static double const zipPar1_[nEta];
  static double const phiPar0_[nEta];
  static double const phiPar1_[nEta];
};

#endif
