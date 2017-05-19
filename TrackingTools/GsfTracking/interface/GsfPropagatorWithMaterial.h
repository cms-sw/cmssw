#ifndef GsfPropagatorWithMaterial_h_
#define GsfPropagatorWithMaterial_h_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/FullConvolutionWithMaterial.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class MagneticField;

// #include "Utilities/Timing/interface/TimingReport.h"

/** \class GsfPropagatorWithMaterial
 * Propagation including material effects on destination surface
 * for multiple trajectory states.
 * Propagates components independently using a specific propagator
 * for the geometrical part and a GsfMaterialEffectsUpdator to include
 * multiple scattering and energy loss at the destination.
 * The number of components will increase according to the result
 * of the GsfMaterialEffectsUpdator.
 */
class GsfPropagatorWithMaterial : public Propagator {

 public:
  // Constructors
  /** Constructor with explicit single state propagator and
   * material effects objects.
   */
  GsfPropagatorWithMaterial(const Propagator& Propagator,
			    const GsfMaterialEffectsUpdator& MEUpdator);
  /** Constructor with explicit multi state propagator and convolutor.
   */
  GsfPropagatorWithMaterial(const GsfPropagatorAdapter& Propagator,
			    const FullConvolutionWithMaterial& Convolutor);

  ~GsfPropagatorWithMaterial() {}


  using Propagator::propagate;
  using Propagator::propagateWithPath;


  /** Propagation to plane with path length calculation.
   */
  virtual std::pair<TrajectoryStateOnSurface,double>
  propagateWithPath (const TrajectoryStateOnSurface&,
		     const Plane&) const override;

  /** Propagation to cylinder with path length calculation.
   */
  virtual std::pair<TrajectoryStateOnSurface,double>
  propagateWithPath (const TrajectoryStateOnSurface&,
		     const Cylinder&) const override;


  /** Propagation to plane with path length calculation.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&,
								   const Plane&) const override;
  /** Propagation to cylinder with path length calculation.
   *  Use from FTS implies single state (better use PropagatorWithMaterial)!
   */
  virtual std::pair<TrajectoryStateOnSurface,double> propagateWithPath (const FreeTrajectoryState&,
								   const Cylinder&) const override;



  virtual bool setMaxDirectionChange( float phiMax) override {
    return theGeometricalPropagator->setMaxDirectionChange(phiMax);
  }

 virtual void setPropagationDirection (PropagationDirection dir) override;

  enum MaterialLocation {atSource, atDestination, fromDirection};
  /** Choice of location for including material effects:
   *  fromDirection is equivalent to atSource for propagation alongMomentum
   *  and to atDestination for propagation oppositeToMomentum.
   *  Inclusion of material effects at the source (either explicitely or
   *  implicitely) is not possible if propagating with anyDirection and
   *  will effectively disable material effects when propagating from
   *  a FreeTrajectoryState.
   */
  void setMaterialLocation (const MaterialLocation location) {
    theMaterialLocation = location;
  }
  /// Access to the geometrical propagator
  const Propagator& geometricalPropagator() const {
    return *theGeometricalPropagator;
  }
  /// Access to the convolutor and thus to the material effects
  const FullConvolutionWithMaterial& convolutionWithMaterial() const {
    return *theConvolutor;
  }

  virtual GsfPropagatorWithMaterial* clone() const override
  {
    return new GsfPropagatorWithMaterial(*theGeometricalPropagator,*theConvolutor);
  }

  const MagneticField* magneticField() const override {return theGeometricalPropagator->magneticField();}

private:
//   /// Definition of timers (temporary)
//   void defineTimer();
  /// Convolution of state+path with material effects
  std::pair<TrajectoryStateOnSurface,double>
  convoluteWithMaterial (const std::pair<TrajectoryStateOnSurface,double>&) const;
  /// Convolution of state with material effects
  TrajectoryStateOnSurface
  convoluteStateWithMaterial (const TrajectoryStateOnSurface, const PropagationDirection) const;
  /// Inclusion of material at the source?
  bool materialAtSource() const;

private:
  // Geometrical propagator
  DeepCopyPointerByClone<GsfPropagatorAdapter> theGeometricalPropagator;
  // Material effects & convolution
  DeepCopyPointerByClone<FullConvolutionWithMaterial> theConvolutor;
  // Use material at source?
  MaterialLocation theMaterialLocation;

  typedef std::pair<TrajectoryStateOnSurface,double> TsosWP;
  typedef std::vector<TrajectoryStateOnSurface> MultiTSOS;

//   static TimingReport::Item* propWithPathTimer1;
//   static TimingReport::Item* propWithPathTimer2;

};

#endif
