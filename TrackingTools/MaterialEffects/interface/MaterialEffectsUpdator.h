#ifndef _CR_MATERIALEFFECTSUPDATOR_H_
#define _CR_MATERIALEFFECTSUPDATOR_H_

/** \class MaterialEffectsUpdator
 *  Interface for adding material effects during propagation.
 *  Updates to TrajectoryStateOnSurface are implemented 
 *  in this class.
 *  Ported from ORCA.
 *
 *  Moved "state" into an independent struct "Effect"
 *
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

namespace materialEffect {
  enum CovIndex { elos = 0, msxx = 1, msxy = 2, msyy = 3 };
  class Covariance {
  public:
    float operator[](CovIndex i) const { return data[i]; }
    float& operator[](CovIndex i) { return data[i]; }
    void add(AlgebraicSymMatrix55& cov) const {
      cov(0, 0) += data[elos];
      cov(1, 1) += data[msxx];
      cov(1, 2) += data[msxy];
      cov(2, 2) += data[msyy];
    }
    Covariance& operator+=(Covariance const& cov) {
      for (int i = 0; i != 4; ++i)
        data[i] += cov.data[i];
      return *this;
    }

  private:
    float data[4] = {0};
  };

  struct Effect {
    float weight = 1.f;
    // Change in |p| from material effects.
    float deltaP = 0;
    // Contribution to covariance matrix (in local co-ordinates) from material effects.
    Covariance deltaCov;
    void combine(Effect const& e1, Effect const& e2) {
      weight *= e1.weight * e2.weight;
      deltaP += e1.deltaP + e2.deltaP;
      deltaCov += e1.deltaCov;
      deltaCov += e2.deltaCov;
    }
  };

}  // namespace materialEffect

class MaterialEffectsUpdator {
public:
  typedef materialEffect::Covariance Covariance;
  typedef materialEffect::Effect Effect;
  typedef materialEffect::CovIndex CovIndex;

  /** Constructor with explicit mass hypothesis
   */
  MaterialEffectsUpdator(float mass);
  virtual ~MaterialEffectsUpdator();

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState(const TrajectoryStateOnSurface& TSoS,
                                               const PropagationDirection propDir) const;

  /** Updates in place TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected)
   *  Will return 'false' if the 'updateState' would have returned an invalid TSOS
   *  Note that the TSoS might be very well unchanged from this method 
   *  (just like 'updateState' can return the same TSOS)
   */
  virtual bool updateStateInPlace(TrajectoryStateOnSurface& TSoS, const PropagationDirection propDir) const;

  /** Particle mass assigned at construction.
   */
  inline float mass() const { return theMass; }

  virtual MaterialEffectsUpdator* clone() const = 0;

  // here comes the actual computation of the values
  virtual void compute(const TrajectoryStateOnSurface&, const PropagationDirection, Effect& effect) const = 0;

private:
  float theMass;
};

#endif
