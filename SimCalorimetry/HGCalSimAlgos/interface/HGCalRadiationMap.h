#ifndef simcalorimetry_hgcalsimalgos_hgcalradiationmap
#define simcalorimetry_hgcalsimalgos_hgcalradiationmap

#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "vdt/vdtMath.h"
#include <string>

/**
   @class HGCalRadiationMap
   @short parses a txt file with dose/fluence parameters and provides functions for noise, etc.
 */
class HGCalRadiationMap {
public:
  struct DoseParameters {
    DoseParameters()
        : a_(0.), b_(0.), c_(0.), d_(0.), e_(0.), doff_(0.), f_(0.), g_(0.), h_(0.), i_(0.), j_(0.), foff_(0.) {}
    double a_, b_, c_, d_, e_, doff_, f_, g_, h_, i_, j_, foff_;
  };

  HGCalRadiationMap();
  ~HGCalRadiationMap(){};

  typedef std::map<std::pair<int, int>, DoseParameters> doseParametersMap;

  void setGeometry(const CaloSubdetectorGeometry *);
  void setDoseMap(const std::string &, const unsigned int);

  double computeRadius(const HGCScintillatorDetId &);

  double getDoseValue(const int, const int, const double, bool logVal = false);
  double getFluenceValue(const int, const int, const double, bool logVal = false);

  const unsigned int &algo() { return algo_; }
  const HGCalGeometry *geom() { return hgcalGeom_; }
  const HGCalTopology *topo() { return hgcalTopology_; }
  const HGCalDDDConstants *ddd() { return hgcalDDD_; }

  inline const doseParametersMap &getDoseMap() { return doseMap_; }
  inline void setFluenceScaleFactor(double val) { fluenceSFlog10_ = log10(val); }

private:
  doseParametersMap readDosePars(const std::string &);

  unsigned int algo_;
  const HGCalGeometry *hgcalGeom_;
  const HGCalTopology *hgcalTopology_;
  const HGCalDDDConstants *hgcalDDD_;
  doseParametersMap doseMap_;
  //conversion from gray to krad (1Gy=100rad=0.1krad)
  const double grayToKrad_ = 0.1;
  double fluenceSFlog10_;
};

#endif
