#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

//
HGCalSciNoiseMap::HGCalSciNoiseMap() : refEdge_(3.) {}

//
void HGCalSciNoiseMap::setSipmMap(const std::string& fullpath) { sipmMap_ = readSipmPars(fullpath); }

//
std::unordered_map<int, float> HGCalSciNoiseMap::readSipmPars(const std::string& fullpath) {
  std::unordered_map<int, float> result;
  //no file means default sipm size
  if (fullpath.empty())
    return result;

  edm::FileInPath fp(fullpath);
  std::ifstream infile(fp.fullPath());
  if (!infile.is_open()) {
    throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
  }
  std::string line;
  while (getline(infile, line)) {
    int layer;
    float boundary;

    //space-separated
    std::stringstream linestream(line);
    linestream >> layer >> boundary;

    result[layer] = boundary;
  }
  return result;
}

//
std::pair<double, double> HGCalSciNoiseMap::scaleByDose(const HGCScintillatorDetId& cellId, const radiiVec& radius) {
  if (getDoseMap().empty())
    return std::make_pair(1., 0.);

  //formula is: A = A0 * exp( -D^0.65 / 199.6)
  //where A0 is the response of the undamaged detector, D is the dose
  int layer = cellId.layer();
  double cellDose = getDoseValue(DetId::HGCalHSc, layer, radius);  //in kRad
  constexpr double expofactor = 1. / 199.6;
  const double dosespower = 0.65;
  double scaleFactor = std::exp(-std::pow(cellDose, dosespower) * expofactor);

  //formula is: N = 2.18 * sqrt(F * A / 2e13)
  //where F is the fluence and A is the SiPM area
  double cellFluence = getFluenceValue(DetId::HGCalHSc, layer, radius);  //in 1-Mev-equivalent neutrons per cm2

  constexpr double fluencefactor = 2. / (2 * 1e13);  //SiPM area = 2mm^2
  const double normfactor = 2.18;
  double noise = normfactor * sqrt(cellFluence * fluencefactor);

  return std::make_pair(scaleFactor, noise);
}

double HGCalSciNoiseMap::scaleByTileArea(const HGCScintillatorDetId& cellId, const radiiVec& radius) {
  double edge;
  if (cellId.type() == 0) {
    constexpr double factor = 2 * M_PI * 1. / 360.;
    edge = radius[0] * factor;  //1 degree
  } else {
    constexpr double factor = 2 * M_PI * 1. / 288.;
    edge = radius[0] * factor;  //1.25 degrees
  }

  double scaleFactor = refEdge_ / edge;  //assume reference 3cm of edge

  return scaleFactor;
}

double HGCalSciNoiseMap::scaleBySipmArea(const HGCScintillatorDetId& cellId, const double& radius) {
  if (sipmMap_.empty())
    return 1.;

  int layer = cellId.layer();
  if (radius < sipmMap_[layer])
    return 2.;
  else
    return 1.;
}

radiiVec HGCalSciNoiseMap::computeRadius(const HGCScintillatorDetId& cellId) {
  GlobalPoint global = geom()->getPosition(cellId);

  double radius2 = std::pow(global.x(), 2) + std::pow(global.y(), 2);  //in cm
  double radius4 = std::pow(radius2, 2);
  double radius = sqrt(radius2);
  double radius3 = radius2 * radius;

  double radius_m100 = radius - 100;
  double radius_m100_2 = std::pow(radius_m100, 2);
  double radius_m100_3 = radius_m100_2 * radius_m100;
  double radius_m100_4 = std::pow(radius_m100_2, 2);

  radiiVec radii{{radius, radius2, radius3, radius4, radius_m100, radius_m100_2, radius_m100_3, radius_m100_4}};
  return radii;
}
