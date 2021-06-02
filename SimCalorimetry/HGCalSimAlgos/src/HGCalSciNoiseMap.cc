#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>

//
HGCalSciNoiseMap::HGCalSciNoiseMap() 
  : refEdge_(3.),
    ignoreSiPMarea_(false), 
    overrideSiPMarea_(false), 
    ignoreTileArea_(false), 
    ignoreDoseScale_(false), 
    ignoreFluenceScale_(false),
    ignoreNoise_(false),
    refDarkCurrent_(0.5)
{
}

//
void HGCalSciNoiseMap::setDoseMap(const std::string &fullpath,const unsigned int algo) {

  //decode bits of the algo word
  ignoreSiPMarea_ = ((algo>>IGNORE_SIPMAREA) & 0x1);
  overrideSiPMarea_ = ((algo>>OVERRIDE_SIPMAREA) & 0x1);
  ignoreTileArea_ = ((algo>>IGNORE_TILEAREA)  & 0x1);
  ignoreDoseScale_ = ((algo>>IGNORE_DOSESCALE) & 0x1);
  ignoreFluenceScale_ = ((algo>>IGNORE_FLUENCESCALE) & 0x1);
  ignoreNoise_ = ((algo>>IGNORE_NOISE) & 0x1);

  //call base class method
  HGCalRadiationMap::setDoseMap(fullpath, algo);
}


//
void HGCalSciNoiseMap::setSipmMap(const std::string& fullpath) { 
  sipmMap_ = readSipmPars(fullpath); 
}

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
void HGCalSciNoiseMap::setReferenceDarkCurrent(double idark) {
  refDarkCurrent_=idark;
}

//
std::pair<double, double> HGCalSciNoiseMap::scaleByDose(const HGCScintillatorDetId& cellId, const double radius) {

  int layer = cellId.layer();
  bool hasDoseMap( !(getDoseMap().empty()));


  //LIGHT YIELD
  double lyScaleFactor(1.f);
  //formula is: A = A0 * exp( -D^0.65 / 199.6)
  //where A0 is the response of the undamaged detector, D is the dose
  if(!ignoreDoseScale_ && hasDoseMap) {
    double cellDose = getDoseValue(DetId::HGCalHSc, layer, radius);  //in kRad
    constexpr double expofactor = 1. / 199.6;
    const double dosespower = 0.65;
    lyScaleFactor = std::exp(-std::pow(cellDose, dosespower) * expofactor);
  }

  //NOISE
  double noise(0.f);
  if(!ignoreNoise_){

    double cellFluence = getFluenceValue(DetId::HGCalHSc, layer, radius);  //in 1-Mev-equivalent neutrons per cm2

    //MODEL 1 : formula is N = 2.18 * sqrt(F * A / 2e13)
    //where F is the fluence and A is the SiPM area (scaling with the latter is done below)
    if(refDarkCurrent_<0) {
      noise = 2.18;
      if (!ignoreFluenceScale_ && hasDoseMap) {
        constexpr double fluencefactor = 2. / (2 * 1e13);  //reference SiPM area = 2mm^2
        noise *= sqrt(cellFluence * fluencefactor);
      }
    }
    
    //MODEL 2 : formula is  3.16 *  sqrt( (Idark * 1e-12) / (qe * gain) * (F / F0) )
    //where F is the fluence (neq/cm2), gain is the SiPM gain, qe is the electron charge (C), Idark is dark current (mA)
    else {
      constexpr double refFluence(2.0E+13);
      constexpr double refGain(235000.);
      double Rdark=(refDarkCurrent_*1E-12)/(CLHEP::e_SI*refGain);
      if(!ignoreFluenceScale_ && hasDoseMap) Rdark *= (cellFluence/refFluence);
      noise = 3.16*sqrt(Rdark);
    }
  }

  //ADDITIONAL SCALING FACTORS
  double tileAreaSF = scaleByTileArea(cellId,radius);
  double sipmAreaSF = scaleBySipmArea(cellId,radius);

  lyScaleFactor *= tileAreaSF*sipmAreaSF;
  noise *= sqrt(sipmAreaSF);

  return std::make_pair(lyScaleFactor, noise);
}

//
double HGCalSciNoiseMap::scaleByTileArea(const HGCScintillatorDetId& cellId, const double radius) {

  double scaleFactor(1.f);

  if(ignoreTileArea_) return scaleFactor;
  
  double edge(refEdge_); //start with reference 3cmm of edge
  if (cellId.type() == 0) {
    constexpr double factor = 2 * M_PI * 1. / 360.;
    edge = radius * factor;  //1 degree
  } else {
    constexpr double factor = 2 * M_PI * 1. / 288.;
    edge = radius * factor;  //1.25 degrees
  }
  scaleFactor = refEdge_ / edge;

  return scaleFactor;
}

//
double HGCalSciNoiseMap::scaleBySipmArea(const HGCScintillatorDetId& cellId, const double radius) {
  
  double scaleFactor(1.f);

  if(ignoreSiPMarea_) return scaleFactor;

  //use sipm area boundary map
  if(overrideSiPMarea_) {
    int layer = cellId.layer();
    if (sipmMap_.count(layer)>0 && radius < sipmMap_[layer]) scaleFactor=2.f;
  }
  //read from DetId
  else {
    int sipm = cellId.sipm(); 
    if( sipm ==1 ) scaleFactor=2.f;
  }

  return scaleFactor;
}
