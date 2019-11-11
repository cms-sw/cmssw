#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

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
double HGCalRadiationMap::getDoseValue(const int subdet, const int layer, const radiiVec& radius, bool logVal) {
  std::pair<int, int> key(subdet, layer);
  double cellDoseLog10 = doseMap_[key].a_ + doseMap_[key].b_ * radius[4] + doseMap_[key].c_ * radius[5] +
                         doseMap_[key].d_ * radius[6] + doseMap_[key].e_ * radius[7];
  return logVal ? cellDoseLog10 * M_LN10 + log(grayToKrad_) : std::pow(10, cellDoseLog10) * grayToKrad_;
}

//
double HGCalRadiationMap::getFluenceValue(const int subdet, const int layer, const radiiVec& radius, bool logVal) {
  std::pair<int, int> key(subdet, layer);
  double cellFluenceLog10 = doseMap_[key].f_ + doseMap_[key].g_ * radius[0] + doseMap_[key].h_ * radius[1] +
                            doseMap_[key].i_ * radius[2] + doseMap_[key].j_ * radius[3];
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
        dosePars.f_ >> dosePars.g_ >> dosePars.h_ >> dosePars.i_ >> dosePars.j_;

    std::pair<int, int> key(subdet, layer);
    result[key] = dosePars;
  }
  return result;
}
