#ifndef SimAlgos_CorrelatedNoisifier_h
#define SimAlgos_CorrelatedNoisifier_h

/**
   \class CorrelatedNoisifier

   \brief adds noise to the given frame

Algorithm taken from
http://cg.scs.carleton.ca/~luc/chapter_eleven.pdf
Uses a Cholesky decomposition
*/
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "DataFormats/Math/interface/Error.h"

#include <valarray>
#include <vector>
#include <utility>
#include <algorithm>
#include <ostream>
#include <cassert>



template<class M> class CorrelatedNoisifier
{
public:

  explicit CorrelatedNoisifier(M & matrix);
  CorrelatedNoisifier(M & matrix, CLHEP::HepRandomEngine & engine);

  virtual ~CorrelatedNoisifier() { delete theRandomGaussian;}

  /// sets all elements along the diagonal of
  /// the correlation matrix to be value
  void setDiagonal(double value);
  
  void setOffDiagonal(int neighbor, double value);

  void setRandomEngine();
  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  template<class T>
  void noisify(T & frame)
  {
    // make a vector of random values
    assert(frame.size() == (int)theSize);
    std::valarray<double> uncorrelated(0.,theSize);
    for (unsigned int i=0; i<theSize; i++)
      uncorrelated[i]=theRandomGaussian->fire();
    if ( isIdentity_ ) 
    {
      for(unsigned int i = 0; i < theSize; ++i)
      { 
        frame[i] += uncorrelated[i]; 
      }
    }
    else if ( isDiagonal_ )
    {
      for(unsigned int i = 0; i < theSize; ++i)
      {
        frame[i] += uncorrelated[i] * std::sqrt(theCovarianceMatrix(i,i)); 
      }
    }
    else 
    {

      // rotate them to make a correlated noise vector
      //std::valarray<double> correlated = theMatrix * uncorrelated;
      std::valarray<double> correlated(0., theSize);
      for (unsigned int i = 0; i < theSize; ++i) 
        {
          //@@ Not sure why the old version only used the lower half, but it worked
          for (unsigned int j = 0; j <= i; ++j) 
            {
              correlated[i] += theMatrix(i,j)*uncorrelated[j];
            }
        }

      // stuff 'em in the frame
      for(unsigned int i = 0; i < theSize; ++i)
      {
        frame[i] += correlated[i];
      }
    }
  }



  void computeDecomposition();

  // for test purpose
  const M & covmatrix() {
    return theCovarianceMatrix;
  }

private:
  void init();
  void checkOffDiagonal();

  const M theCovarianceMatrix;
  M theMatrix;
  mutable RandGaussQ * theRandomGaussian;
  unsigned int theSize; 
  bool isDiagonal_;
  bool isIdentity_;

};

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

template<class M> CorrelatedNoisifier<M>::CorrelatedNoisifier(M & matrix)
  :  theCovarianceMatrix(matrix),
     theRandomGaussian(0),
     theSize(matrix.kRows),
     isDiagonal_(true),
     isIdentity_(true)
{
  init();
  setRandomEngine();
}
 

template<class M> CorrelatedNoisifier<M>::CorrelatedNoisifier(M & matrix, CLHEP::HepRandomEngine & engine)
  :  theCovarianceMatrix(matrix),
     theRandomGaussian(0),
     theSize(matrix.kRows),
     isDiagonal_(true),
     isIdentity_(true)
{
  init();
  setRandomEngine(engine);
}


template<class M>
void CorrelatedNoisifier<M>::init()
{
  for ( unsigned int i = 0 ; i < theSize ; i++ ) {
    theMatrix(i,i) = 0.;
    for ( unsigned int j = 0 ; j < theSize ; j++ ) {
      theMatrix(i,j) = 0.;
      theMatrix(j,i) = 0.;
    }
  }

  checkOffDiagonal();

  if ( ! isDiagonal_ ) computeDecomposition();
}


template<class M> void CorrelatedNoisifier<M>::setRandomEngine()
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


template<class M> void CorrelatedNoisifier<M>::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  if(theRandomGaussian) delete theRandomGaussian;
  theRandomGaussian = new CLHEP::RandGaussQ(engine);
}


template<class M> void CorrelatedNoisifier<M>::setDiagonal(double value) 
{
  for(unsigned int i = 0; i < theSize; ++i) 
  {
    theCovarianceMatrix(i,i) = value;
  }

  checkOffDiagonal();

  if ( ! isDiagonal_ ) computeDecomposition();

} 

template<class M> void CorrelatedNoisifier<M>::setOffDiagonal(int distance, double value)
{
  for(unsigned int column = 0; column < theSize; ++column)
  {
    unsigned int row = column - distance;
    if(row < 0) continue;
    theCovarianceMatrix(row,column) = value;
    theCovarianceMatrix(column,row) = value;

  }

  checkOffDiagonal();

  if ( ! isDiagonal_ ) computeDecomposition();

}


template<class M> void CorrelatedNoisifier<M>::computeDecomposition()
{

  for ( unsigned int i = 0 ; i < theSize ; i++ ) {
    for ( unsigned int j = 0 ; j < theSize ; j++ ) {
      theMatrix(i,j) = 0.;
    }
  }

  double sqrtSigma00 = theCovarianceMatrix(0,0);
  if ( sqrtSigma00 <= 0. ) {
    throw cms::Exception("CorrelatedNoisifier") << "non positive variance.";
  }
  sqrtSigma00 = std::sqrt(sqrtSigma00);

  for ( unsigned int i = 0 ; i < theSize ; i++ )
    {
      double hi0 = theCovarianceMatrix(i,0)/sqrtSigma00;
      theMatrix(i,0) = hi0;
    }

  for ( unsigned int i = 1 ; i < theSize ; i++ ) 
    {

      for ( unsigned int j = 1 ; j < i ; j++ )
        {
          double hij = theCovarianceMatrix(i,j);
          for ( unsigned int k = 0 ; k <= j-1 ; k++ ) hij -= theMatrix(i,k)*theMatrix(j,k);
          hij /= theMatrix(j,j);
          theMatrix(i,j) = hij;
        }
      
      double hii = theCovarianceMatrix(i,i);
      for ( unsigned int j = 0 ; j <= i-1 ; j++ ) {
        double hij = theMatrix(i,j);
        hii -= hij*hij;
      }
      hii = sqrt(hii);
      theMatrix(i,i) = hii;

    }
}

template<class M>
void CorrelatedNoisifier<M>::checkOffDiagonal() {

  isDiagonal_ = true;
  isIdentity_ = true;

  for ( unsigned int i = 0 ; i < theSize ; i++ ) {
    for ( unsigned int j = 0 ; j < theSize ; j++ ) {
      if ( i == j )
      {
        if( theCovarianceMatrix(i,j) != 1.)
        {
          isIdentity_ = false;
        }
        if( theCovarianceMatrix(i,j) < 0.)
        {
          throw cms::Exception("CorrelatedNoisifier") 
            << "Bad correlation matrix.  Negative diagonal";
        }
      }
      if ( i != j && theCovarianceMatrix(i,j) != 0. ) 
      { 
        isDiagonal_ = false ; 
        isIdentity_ = false;
        return ; 
      }
    }
  }
  
}

#endif
