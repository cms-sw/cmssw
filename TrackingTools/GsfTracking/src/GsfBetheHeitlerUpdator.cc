#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"

#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <fstream>
#include <cmath>

GsfBetheHeitlerUpdator::GsfBetheHeitlerUpdator(const std::string fileName,
					       const int correctionFlag) :
  // 					       const CorrectionFlag correctionFlag) :
  GsfMaterialEffectsUpdator(0.000511),
  theNrComponents(0),
  theCorrectionFlag(correctionFlag),
  theLastDz(0.),
  theLastP(-1.),
  theLastPropDir(alongMomentum),
  theLastRadLength(-1.)
{
  if ( theCorrectionFlag==1 )
    edm::LogInfo("GsfBetheHeitlerUpdator") << "1st moment of mixture will be corrected";
  if ( theCorrectionFlag>=2 )
    edm::LogInfo("GsfBetheHeitlerUpdator")
      << "1st and 2nd moments of mixture will be corrected";

  readParameters(fileName);
}

void GsfBetheHeitlerUpdator::readParameters (const std::string fileName)
{  
  std::string name = "TrackingTools/GsfTracking/data/";
  name += fileName;
  
  edm::FileInPath parFile(name);
  edm::LogInfo("GsfBetheHeitlerUpdator") << "Reading GSF parameterization " 
					 << "of Bethe-Heitler energy loss from "
					 << parFile.fullPath();
  std::ifstream ifs(parFile.fullPath().c_str());

  ifs >> theNrComponents;
  int orderP;
  ifs >> orderP;
  ifs >> theTransformationCode;
  for ( int ic=0; ic<theNrComponents; ic++ ) {
    thePolyWeights.push_back(readPolynomial(ifs,orderP));
    thePolyMeans.push_back(readPolynomial(ifs,orderP));
    thePolyVars.push_back(readPolynomial(ifs,orderP));
  }
}

GsfBetheHeitlerUpdator::Polynomial
GsfBetheHeitlerUpdator::readPolynomial (std::ifstream& aStream, 
					const int order) {
  std::vector<double> coeffs(order+1);
  for ( int i=0; i<(order+1); i++ ) aStream >> coeffs[i];
  return Polynomial(coeffs);
}

void
GsfBetheHeitlerUpdator::compute (const TrajectoryStateOnSurface& TSoS,
				 const PropagationDirection propDir) const 
{
  //
  // clear cache
  //
  theWeights.clear();
  theDeltaPs.clear();
  theDeltaCovs.clear();
  //
  // Get surface and check presence of medium properties
  //
  const Surface& surface = TSoS.surface();
  //
  // calculate components: first check associated material constants
  //
  double rl(0.);
  double p(0.);
  if ( surface.mediumProperties() ) {
    LocalVector pvec = TSoS.localMomentum();
    p = pvec.mag();
    rl = surface.mediumProperties()->radLen()/fabs(pvec.z())*p;
  }
  //
  // produce multi-state only in case of x/X0>0
  //
  if ( rl>0.0001 ) {
    //
    // limit x/x0 to valid range for parametrisation
    // should be done in a more elegant way ...
    //
    if ( rl<0.01 )  rl = 0.01;
    if ( rl>0.20 )  rl = 0.20;

    GSContainer mixture;
    getMixtureParameters(rl,mixture);
    correctWeights(mixture);
    if ( theCorrectionFlag>=1 )
      mixture[0].second = correctedFirstMean(rl,mixture);
    if ( theCorrectionFlag>=2 )
      mixture[0].third = correctedFirstVar(rl,mixture);

    for ( int i=0; i<theNrComponents; i++ ) {
      double varPinv;
      theWeights.push_back(mixture[i].first);
      if ( propDir==alongMomentum ) {
	//
	// for forward propagation: calculate in p (linear in 1/z=p_inside/p_outside),
	// then convert sig(p) to sig(1/p). 
	//
	theDeltaPs.push_back(p*(mixture[i].second-1));
	//    double f = 1./p/mixture[i].second/mixture[i].second;
	// patch to ensure consistency between for- and backward propagation
	double f = 1./p/mixture[i].second;
	varPinv = f*f*mixture[i].third;
      }
      else {
	//
	// for backward propagation: delta(1/p) is linear in z=p_outside/p_inside
	// convert to obtain equivalent delta(p)
	//
	theDeltaPs.push_back(p*(1/mixture[i].second-1));
	varPinv = mixture[i].third/p/p;
      }
      AlgebraicSymMatrix55 errors;
      errors(0,0) = varPinv;
      theDeltaCovs.push_back(errors);
    }
  }
  else {
    theWeights.push_back(1.);
    theDeltaPs.push_back(0.);
    theDeltaCovs.push_back(AlgebraicSymMatrix55());
  }
  //
  // Save arguments to avoid duplication of computation
  //
  storeArguments(TSoS,propDir); 
}
//
// Mixture parameters (in z)
//
void 
GsfBetheHeitlerUpdator::getMixtureParameters (const double rl,
					      GSContainer& mixture) const
{
  mixture.clear();
  mixture.reserve(theNrComponents);

  for ( int i=0; i<theNrComponents; i++ ) {

    double weight = thePolyWeights[i](rl);
    if ( theTransformationCode )  weight = logisticFunction(weight);

    double z = thePolyMeans[i](rl);
    if ( theTransformationCode )  z = logisticFunction(z);

    double vz = thePolyVars[i](rl);
    if ( theTransformationCode )  vz = exp(vz);
    else                          vz = vz*vz;

    mixture.push_back(Triplet<double,double,double>(weight,z,vz));
  }
}

//
// Correct weights
//
void
GsfBetheHeitlerUpdator::correctWeights (GSContainer& mixture) const
{
  if ( mixture.empty() )  return;
  //
  // get sum of weights
  //
  double wsum(0);
  for ( GSContainer::const_iterator i=mixture.begin();
	i!=mixture.end(); i++ )  wsum += (*i).first;
  //
  // rescale to obtain 1
  //
  for ( GSContainer::iterator i=mixture.begin();
	i!=mixture.end(); i++ )  (*i).first /= wsum;
}
//
// Correct means
//
double
GsfBetheHeitlerUpdator::correctedFirstMean (const double rl,
					    const GSContainer& mixture) const
{
  if ( mixture.empty() )  return 0.;
  //
  // calculate difference true mean - weighted sum
  //
  double mean = BetheHeitlerMean(rl);
  for ( GSContainer::const_iterator i=mixture.begin()+1;
	i!=mixture.end(); i++ )  mean -= (*i).first*(*i).second;
  //
  // return corrected mean for first component
  //
  return std::max(std::min(mean/mixture[0].first,1.),0.);
}
//
// Correct variances
//
double
GsfBetheHeitlerUpdator::correctedFirstVar (const double rl,
					   const GSContainer& mixture) const
{
  if ( mixture.empty() )  return 0.;
  //
  // calculate difference true variance - weighted sum
  //
  double var = BetheHeitlerVariance(rl) +
    BetheHeitlerMean(rl)*BetheHeitlerMean(rl) -
    mixture[0].first*mixture[0].second*mixture[0].second;
  for ( GSContainer::const_iterator i=mixture.begin()+1;
	i!=mixture.end(); i++ )
    var -= (*i).first*((*i).second*(*i).second+(*i).third);
  //
  // return corrected variance for first component
  //
  return std::max(var/mixture[0].first,0.);
}

//
// Compare arguments with the ones of the previous call
//
bool 
GsfBetheHeitlerUpdator::newArguments (const TrajectoryStateOnSurface& TSoS, 
				      const PropagationDirection propDir) const {
  return TSoS.localMomentum().unit().z()!=theLastDz ||
    TSoS.localMomentum().mag()!=theLastP || propDir!=theLastPropDir ||
    TSoS.surface().mediumProperties()->radLen()!=theLastRadLength;
}
//
// Save arguments
//
void GsfBetheHeitlerUpdator::storeArguments (const TrajectoryStateOnSurface& TSoS, 
					     const PropagationDirection propDir) const {
  theLastDz = TSoS.localMomentum().unit().z();
  theLastP = TSoS.localMomentum().mag();
  theLastPropDir = propDir;
  theLastRadLength = TSoS.surface().mediumProperties()->radLen();
}
