#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

//
HGCalRadiationMap::HGCalRadiationMap() : fluenceSFlog10_(0.) {}

//
void HGCalRadiationMap::setDoseMap(const std::string& fullpath, const unsigned int algo) {
  doseMap_ = readDosePars(fullpath);
  algo_ = algo;
}

//
void HGCalRadiationMap::setGeometry(const CaloSubdetectorGeometry* geom) {
  hgcalGeom_ = static_cast<const HGCalGeometry*>(geom);
  hgcalTopology_ = &(hgcalGeom_->topology());
  hgcalDDD_ = &(hgcalTopology_->dddConstants());
}

//
double HGCalRadiationMap::computeRadius(const HGCScintillatorDetId& cellId) {
  GlobalPoint global = geom()->getPosition(cellId);
  return std::sqrt(std::pow(global.x(), 2) + std::pow(global.y(), 2));
}

//
double HGCalRadiationMap::getDoseValue(const int subdet, const int layer, const double radius, bool logVal) {
  std::pair<int, int> key(subdet, layer);

  if (doseMap_.find(key) == doseMap_.end()) {
    return logVal ? -10. : 0.;
  }

  double r(radius - doseMap_[key].doff_);
  double r2(r * r);
  double r3(r2 * r);
  double r4(r3 * r);

  double cellDoseLog10 =
      doseMap_[key].a_ + doseMap_[key].b_ * r + doseMap_[key].c_ * r2 + doseMap_[key].d_ * r3 + doseMap_[key].e_ * r4;

  return logVal ? cellDoseLog10 * M_LN10 + log(grayToKrad_) : std::pow(10, cellDoseLog10) * grayToKrad_;
}

//
double HGCalRadiationMap::getFluenceValue(const int subdet, const int layer, const double radius, bool logVal) {
  std::pair<int, int> key(subdet, layer);

  double r(radius - doseMap_[key].foff_);
  double r2(r * r);
  double r3(r2 * r);
  double r4(r3 * r);

  double cellFluenceLog10 =
      doseMap_[key].f_ + doseMap_[key].g_ * r + doseMap_[key].h_ * r2 + doseMap_[key].i_ * r3 + doseMap_[key].j_ * r4;
  cellFluenceLog10 += fluenceSFlog10_;

  return logVal ? cellFluenceLog10 * M_LN10 : std::pow(10, cellFluenceLog10);
}

//
std::map<std::pair<int, int>, HGCalRadiationMap::DoseParameters> HGCalRadiationMap::readDosePars(
    const std::string& fullpath) {
  doseParametersMap result;

  //no dose file means no aging
  if (fullpath.empty())
    return result;

  edm::FileInPath fp(fullpath);
  std::ifstream infile(fp.fullPath());
  if (!infile.is_open()) {
    throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
  }
  std::string line;
  while (getline(infile, line)) {
    int subdet;
    int layer;
    DoseParameters dosePars;

    //space-separated
    std::stringstream linestream(line);
    linestream >> subdet >> layer >> dosePars.a_ >> dosePars.b_ >> dosePars.c_ >> dosePars.d_ >> dosePars.e_ >>
        dosePars.doff_ >> dosePars.f_ >> dosePars.g_ >> dosePars.h_ >> dosePars.i_ >> dosePars.j_ >> dosePars.foff_;

    std::pair<int, int> key(subdet, layer);
    result[key] = dosePars;
  }
  return result;
}
