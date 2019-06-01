#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"

#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <fstream>
#include <cmath>

#include <cassert>

namespace {
  /// Logistic function (needed for transformation of weight and mean)
  inline float logisticFunction(const float x) { return 1. / (1. + unsafe_expf<4>(-x)); }
  /// First moment of the Bethe-Heitler distribution (in z=E/E0)
  inline float BetheHeitlerMean(const float rl) { return unsafe_expf<4>(-rl); }
  /// Second moment of the Bethe-Heitler distribution (in z=E/E0)
  inline float BetheHeitlerVariance(const float rl) {
#if defined(__clang__) || defined(__INTEL_COMPILER)
    const
#else
    constexpr
#endif
        float l3ol2 = std::log(3.) / std::log(2.);
    float mean = BetheHeitlerMean(rl);
    return unsafe_expf<4>(-rl * l3ol2) - mean * mean;
  }
}  // namespace

/*
namespace {
 /// Logistic function (needed for transformation of weight and mean)
  inline float logisticFunction (const float x) {return 1.f/(1.f+std::exp(-x));}
  /// First moment of the Bethe-Heitler distribution (in z=E/E0)
  inline float BetheHeitlerMean (const float rl) {
    return std::exp(-rl);
  }
  /// Second moment of the Bethe-Heitler distribution (in z=E/E0)
  inline float BetheHeitlerVariance (const float rl)
  {
#if __clang__
    const
#else    
    constexpr
#endif
    float l3ol2 = std::log(3.)/std::log(2.);
    return std::exp(-rl*l3ol2) -  std::exp(-2*rl);
  }
}
*/

GsfBetheHeitlerUpdator::GsfBetheHeitlerUpdator(const std::string fileName, const int correctionFlag)
    : GsfMaterialEffectsUpdator(0.000511, 6), theNrComponents(0), theCorrectionFlag(correctionFlag) {
  if (theCorrectionFlag == 1)
    edm::LogInfo("GsfBetheHeitlerUpdator") << "1st moment of mixture will be corrected";
  if (theCorrectionFlag >= 2)
    edm::LogInfo("GsfBetheHeitlerUpdator") << "1st and 2nd moments of mixture will be corrected";

  readParameters(fileName);
  assert(theNrComponents <= 6);
  resize(theNrComponents);
}

void GsfBetheHeitlerUpdator::readParameters(const std::string fileName) {
  std::string name = "TrackingTools/GsfTracking/data/";
  name += fileName;

  edm::FileInPath parFile(name);
  edm::LogInfo("GsfBetheHeitlerUpdator") << "Reading GSF parameterization "
                                         << "of Bethe-Heitler energy loss from " << parFile.fullPath();
  std::ifstream ifs(parFile.fullPath().c_str());

  ifs >> theNrComponents;
  int orderP;
  ifs >> orderP;
  ifs >> theTransformationCode;

  assert(orderP < MaxOrder);

  for (int ic = 0; ic != theNrComponents; ++ic) {
    thePolyWeights[ic] = readPolynomial(ifs, orderP);
    thePolyMeans[ic] = readPolynomial(ifs, orderP);
    thePolyVars[ic] = readPolynomial(ifs, orderP);
  }
}

GsfBetheHeitlerUpdator::Polynomial GsfBetheHeitlerUpdator::readPolynomial(std::ifstream& aStream,
                                                                          const unsigned int order) {
  float coeffs[order + 1];
  for (unsigned int i = 0; i < (order + 1); ++i)
    aStream >> coeffs[i];
  return Polynomial(coeffs, order + 1);
}

void GsfBetheHeitlerUpdator::compute(const TrajectoryStateOnSurface& TSoS,
                                     const PropagationDirection propDir,
                                     Effect effects[]) const {
  //
  // Get surface and check presence of medium properties
  //
  const Surface& surface = TSoS.surface();
  //
  // calculate components: first check associated material constants
  //
  float rl(0.f);
  float p(0.f);
  if (surface.mediumProperties().isValid()) {
    LocalVector pvec = TSoS.localMomentum();
    p = pvec.mag();
    rl = surface.mediumProperties().radLen() / fabs(pvec.z()) * p;
  }
  //
  // produce multi-state only in case of x/X0>0
  //
  if (rl > 0.0001f) {
    //
    // limit x/x0 to valid range for parametrisation
    // should be done in a more elegant way ...
    //
    if (rl < 0.01f)
      rl = 0.01f;
    if (rl > 0.20f)
      rl = 0.20f;

    float mixtureData[3][theNrComponents];
    GSContainer mixture{mixtureData[0], mixtureData[1], mixtureData[2]};

    getMixtureParameters(rl, mixture);
    correctWeights(mixture);
    if (theCorrectionFlag >= 1)
      mixture.second[0] = correctedFirstMean(rl, mixture);
    if (theCorrectionFlag >= 2)
      mixture.third[0] = correctedFirstVar(rl, mixture);

    for (int i = 0; i < theNrComponents; i++) {
      float varPinv;
      effects[i].weight *= mixture.first[i];
      if (propDir == alongMomentum) {
        //
        // for forward propagation: calculate in p (linear in 1/z=p_inside/p_outside),
        // then convert sig(p) to sig(1/p).
        //
        effects[i].deltaP += p * (mixture.second[i] - 1.f);
        //    float f = 1./p/mixture.second[i]/mixture.second[i];
        // patch to ensure consistency between for- and backward propagation
        float f = 1.f / (p * mixture.second[i]);
        varPinv = f * f * mixture.third[i];
      } else {
        //
        // for backward propagation: delta(1/p) is linear in z=p_outside/p_inside
        // convert to obtain equivalent delta(p)
        //
        effects[i].deltaP += p * (1.f / mixture.second[i] - 1.f);
        varPinv = mixture.third[i] / (p * p);
      }
      using namespace materialEffect;
      effects[i].deltaCov[elos] += varPinv;
    }
  }
}
//
// Mixture parameters (in z)
//
void GsfBetheHeitlerUpdator::getMixtureParameters(const float rl, GSContainer& mixture) const {
  float weight[theNrComponents], z[theNrComponents], vz[theNrComponents];
  for (int i = 0; i < theNrComponents; i++) {
    weight[i] = thePolyWeights[i](rl);
    z[i] = thePolyMeans[i](rl);
    vz[i] = thePolyVars[i](rl);
  }
  if (theTransformationCode)
    for (int i = 0; i < theNrComponents; i++) {
      mixture.first[i] = logisticFunction(weight[i]);
      mixture.second[i] = logisticFunction(z[i]);
      mixture.third[i] = unsafe_expf<4>(vz[i]);
      ;
    }
  else  // theTransformationCode
    for (int i = 0; i < theNrComponents; i++) {
      mixture.first[i] = weight[i];
      mixture.second[i] = z[i];
      mixture.third[i] = vz[i] * vz[i];
    }
}

//
// Correct weights
//
void GsfBetheHeitlerUpdator::correctWeights(GSContainer& mixture) const {
  //
  // get sum of weights
  //
  float wsum(0);
  for (int i = 0; i < theNrComponents; i++)
    wsum += mixture.first[i];
  //
  // rescale to obtain 1
  //
  wsum = 1.f / wsum;
  for (int i = 0; i < theNrComponents; i++)
    mixture.first[i] *= wsum;
}
//
// Correct means
//
float GsfBetheHeitlerUpdator::correctedFirstMean(const float rl, const GSContainer& mixture) const {
  //
  // calculate difference true mean - weighted sum
  //
  float mean = BetheHeitlerMean(rl);
  for (int i = 1; i < theNrComponents; i++)
    mean -= mixture.first[i] * mixture.second[i];
  //
  // return corrected mean for first component
  //
  return std::max(std::min(mean / mixture.first[0], 1.f), 0.f);
}
//
// Correct variances
//
float GsfBetheHeitlerUpdator::correctedFirstVar(const float rl, const GSContainer& mixture) const {
  //
  // calculate difference true variance - weighted sum
  //
  float var = BetheHeitlerVariance(rl) + BetheHeitlerMean(rl) * BetheHeitlerMean(rl) -
              mixture.first[0] * mixture.second[0] * mixture.second[0];
  for (int i = 1; i < theNrComponents; i++)
    var -= mixture.first[i] * (mixture.second[i] * mixture.second[i] + mixture.third[i]);
  //
  // return corrected variance for first component
  //
  return std::max(var / mixture.first[0], 0.f);
}
