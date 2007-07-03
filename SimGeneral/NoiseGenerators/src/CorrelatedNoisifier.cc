#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"


CorrelatedNoisifier::CorrelatedNoisifier(int nFrames)
: theCovarianceMatrix(nFrames, 1.0),
  theMatrix(nFrames, 1.0),
  theRandomGaussian(0),
  theSize(nFrames)
{

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();
  setRandomEngine();
}


CorrelatedNoisifier::CorrelatedNoisifier(const HepSymMatrix & matrix)
: theCovarianceMatrix(matrix.num_row(),matrix),
  theMatrix(matrix.num_row(),1.0),
  theRandomGaussian(0),
  theSize(theCovarianceMatrix.rank())
{

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();
  setRandomEngine();
}


void CorrelatedNoisifier::initializeServices()
{
  if(not edmplugin::PluginManager::isAvailable()) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

  std::string config =
  "process CorrNoise = {"
    "service = RandomNumberGeneratorService"
    "{"
      "untracked uint32 sourceSeed = 123456789"
    "}"
  "}";

  //create the services
  edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

  //make the services available
  edm::ServiceRegistry::Operate operate(tempToken);
}

 
void CorrelatedNoisifier::setRandomEngine()
{
   edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "CorrelatedNoisifier requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }
   setRandomEngine(rng->getEngine());
}


void CorrelatedNoisifier::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  if(theRandomGaussian) delete theRandomGaussian;
  theRandomGaussian = new CLHEP::RandGaussQ(engine);
}


void CorrelatedNoisifier::setDiagonal(double value) 
{
  for(int i = 0; i < theSize; ++i) 
  {
    theCovarianceMatrix(i,i) = value;
  }

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

} 

void CorrelatedNoisifier::setOffDiagonal(int distance, double value)
{
  for(int column = 0; column < theSize; ++column)
  {
    int row = column - distance;
    if(row < 0) continue;
    theCovarianceMatrix(row,column) = value;
    theCovarianceMatrix(column,row) = value;

  }

  isDiagonal_ = true;
  checkOffDiagonal(isDiagonal_);

  computeDecomposition();

}


void CorrelatedNoisifier::computeDecomposition()
{

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {
      theMatrix(i,j) = 0.;
    }
  }

  double sqrtSigma00 = theCovarianceMatrix(0,0);
  if ( sqrtSigma00 <= 0. ) {
    throw cms::Exception("CorrelatedNoisifier") << "non positive variance.";
  }
  sqrtSigma00 = std::sqrt(sqrtSigma00);

  for ( int i = 0 ; i < theSize ; i++ )
    {
      double hi0 = theCovarianceMatrix(i,0)/sqrtSigma00;
      theMatrix(i,0) = hi0;
    }

  for ( int i = 1 ; i < theSize ; i++ ) 
    {

      for ( int j = 1 ; j < i ; j++ )
        {
          double hij = theCovarianceMatrix(i,j);
          for ( int k = 0 ; k <= j-1 ; k++ ) hij -= theMatrix(i,k)*theMatrix(j,k);
          hij /= theMatrix(j,j);
          theMatrix(i,j) = hij;
        }
      
      double hii = theCovarianceMatrix(i,i);
      for ( int j = 0 ; j <= i-1 ; j++ ) {
        double hij = theMatrix(i,j);
        hii -= hij*hij;
      }
      hii = sqrt(hii);
      theMatrix(i,i) = hii;

    }

}

void CorrelatedNoisifier::checkOffDiagonal(bool & isDiagonal_) const {

  isDiagonal_ = true;

  for ( int i = 0 ; i < theSize ; i++ ) {
    for ( int j = 0 ; j < theSize ; j++ ) {

      if ( i != j && theCovarianceMatrix(i,j) != 0. ) { isDiagonal_ = false ; return ; }
      
    }
  }
  

}
