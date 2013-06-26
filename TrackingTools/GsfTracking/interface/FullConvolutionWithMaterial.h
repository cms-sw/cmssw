#ifndef FullConvolutionWithMaterial_h_
#define FullConvolutionWithMaterial_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

//  #include "Utilities/Notification/interface/TimingReport.h"

/** \class FullConvolutionWithMaterial
 *  Convolute a set of trajectory states with
 *  material effects. 
 */
class FullConvolutionWithMaterial {  
  
public:
  /// Constructor with GSF material effects updator and propagation direction.
  FullConvolutionWithMaterial(const GsfMaterialEffectsUpdator& aMEUpdator) :
    theMEUpdator(aMEUpdator.clone()) {}

  ~FullConvolutionWithMaterial() {};

  /// Convolution using the GsfMaterialEffectsUpdator
  TrajectoryStateOnSurface operator() (const TrajectoryStateOnSurface&,
				       const PropagationDirection) const;

  /// Access to material effects updator
  inline const GsfMaterialEffectsUpdator& materialEffectsUpdator () const
  {
    return *theMEUpdator;
  }

  /// Clone
  FullConvolutionWithMaterial* clone() const
  {
    return new FullConvolutionWithMaterial(*this);
  }

private:
  // Material effects
  DeepCopyPointerByClone<GsfMaterialEffectsUpdator> theMEUpdator;
  
//    static TimingReport::Item* theTimer1;
//    static TimingReport::Item* theTimer2;
  
};
#endif
