#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoisifier.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/JamesRandom.h" 
#include "CLHEP/Random/RandGaussQ.h"

namespace cms {

  CaloVNoisifier::CaloVNoisifier() :
     theRandGaussian(new CLHEP::RandGaussQ(*CLHEP::HepRandom::getTheEngine()))
  {
  }

  CaloVNoisifier::~CaloVNoisifier() {
    delete theRandGaussian;
  }

}

