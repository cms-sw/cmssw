#include "TrackingTools/GsfTracking/interface/GsfCombinedMaterialEffectsUpdator.h"
#include<cassert>

#include<iostream>

GsfCombinedMaterialEffectsUpdator::GsfCombinedMaterialEffectsUpdator(GsfMaterialEffectsUpdator& msUpdator,
								     GsfMaterialEffectsUpdator& elUpdator) :
  GsfMaterialEffectsUpdator(msUpdator.mass(),msUpdator.size()*elUpdator.size()),
  theMSUpdator(msUpdator.clone()), 
  theELUpdator(elUpdator.clone()) {}

//
// Computation: combine updates on momentum and cov. matrix from the multiple
// scattering and energy loss updators and store them in private data members
//
void GsfCombinedMaterialEffectsUpdator::compute (const TrajectoryStateOnSurface& TSoS,
						 const PropagationDirection propDir, Effect effects[]) const
{
  #if __clang__
  std::vector<Effect> msEffects(theMSUpdator->size());
  theMSUpdator->compute(TSoS,propDir,msEffects.data());
  std::vector<Effect> elEffects(theELUpdator->size());
  theELUpdator->compute(TSoS,propDir,elEffects.data());
  #else
  Effect msEffects[theMSUpdator->size()];
  theMSUpdator->compute(TSoS,propDir,msEffects);
  Effect elEffects[theELUpdator->size()];
  theELUpdator->compute(TSoS,propDir,elEffects);
  #endif
 
  //
  // combine the two multi-updates
  //
  uint32_t k=0;
  for ( auto const & mse : msEffects ) 
    for ( auto const & ele : elEffects)
      effects[k++].combine(mse,ele);
  assert(k==size());
}

