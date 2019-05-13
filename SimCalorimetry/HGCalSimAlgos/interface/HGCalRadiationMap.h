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
class HGCalRadiationMap
{
  public:

  struct DoseParameters {
  DoseParameters(): a_(0.), b_(0.), c_(0.), d_(0.), e_(0.),
      f_(0.), g_(0.), h_(0.), i_(0.), j_(0.) {}
      double a_, b_, c_, d_, e_, f_, g_, h_, i_, j_;
  };
  
  HGCalRadiationMap(); 
  ~HGCalRadiationMap() {};
  
  void setGeometry(const CaloSubdetectorGeometry*);
  void setDoseMap(const std::string&);
  
  double scaleByArea(const HGCScintillatorDetId&, const std::array<double, 8>&);
  std::pair<double, double> scaleByDose(const HGCScintillatorDetId&, const std::array<double, 8>&);
  double getDoseValue(const int, const int, const std::array<double, 8>&,bool logVal=false);
  double getFluenceValue(const int, const int, const std::array<double, 8>&,bool logVal=false);
  std::array<double, 8> computeRadius(const HGCScintillatorDetId&);

  const HGCalGeometry *geom()    { return hgcalGeom_; }
  const HGCalDDDConstants *ddd() { return hgcalDDD_; }
  inline const std::map<std::pair<int,int>, DoseParameters> & getDoseMap() { return doseMap_; }
  
 private:
  std::map<std::pair<int,int>, DoseParameters> readDosePars(const std::string&);
  
  const HGCalGeometry *hgcalGeom_;
  const HGCalDDDConstants *hgcalDDD_;
  std::map<std::pair<int,int>, DoseParameters> doseMap_;
  const double grayToKrad_;
  const double refEdge_; //cm
  bool verbose_;
  
};

#endif
