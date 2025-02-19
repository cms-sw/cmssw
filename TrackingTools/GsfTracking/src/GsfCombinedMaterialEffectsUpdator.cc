#include "TrackingTools/GsfTracking/interface/GsfCombinedMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsAdapter.h"
// #include "CommonReco/MaterialEffects/interface/MultipleScatteringUpdator.h"
// #include "CommonReco/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"

// GsfCombinedMaterialEffectsUpdator::GsfCombinedMaterialEffectsUpdator () :
//   GsfMaterialEffectsUpdator()
// {
//   createUpdators(mass());
// }

// GsfCombinedMaterialEffectsUpdator::GsfCombinedMaterialEffectsUpdator (float mass) :
//   GsfMaterialEffectsUpdator(mass)
// {
//   createUpdators(mass);
// }

GsfCombinedMaterialEffectsUpdator::GsfCombinedMaterialEffectsUpdator  
(GsfMaterialEffectsUpdator& msUpdator,
 GsfMaterialEffectsUpdator& elUpdator) :
  GsfMaterialEffectsUpdator(msUpdator.mass()),
  theMSUpdator(msUpdator.clone()), 
  theELUpdator(elUpdator.clone()) {}

// void
// GsfCombinedMaterialEffectsUpdator::createUpdators (const float mass)
// {
//   //
//   // multiple scattering
//   //
//   theMSUpdator = new GsfMaterialEffectsAdapter(MultipleScatteringUpdator(mass));
//   //
//   // energy loss: two different objects for electrons / others
//   //
//   if ( mass>0.001 ) 
//     theELUpdator = new GsfMaterialEffectsAdapter(EnergyLossUpdator(mass));
//   else
//     theELUpdator = new GsfBetheHeitlerUpdator();
// }

//
// Computation: combine updates on momentum and cov. matrix from the multiple
// scattering and energy loss updators and store them in private data members
//
void GsfCombinedMaterialEffectsUpdator::compute (const TrajectoryStateOnSurface& TSoS,
						 const PropagationDirection propDir) const
{
  //
  // reset components
  //
  theWeights.clear();
  theDeltaPs.clear();
  theDeltaCovs.clear();
  //
  // get components from multiple scattering
  //
  std::vector<double> msWeights = theMSUpdator->weights(TSoS,propDir);
  std::vector<double> msDeltaPs = theMSUpdator->deltaPs(TSoS,propDir);
  std::vector<AlgebraicSymMatrix55> msDeltaCovs = theMSUpdator->deltaLocalErrors(TSoS,propDir);
  if ( msWeights.empty() ) {
    //
    // create one dummy component
    //
    msWeights.push_back(1.);
    msDeltaPs.push_back(0.);
    msDeltaCovs.push_back(AlgebraicSymMatrix55());
  }
  //
  // get components from energy loss
  //
  std::vector<double> elWeights = theELUpdator->weights(TSoS,propDir);
  std::vector<double> elDeltaPs = theELUpdator->deltaPs(TSoS,propDir);
  std::vector<AlgebraicSymMatrix55> elDeltaCovs = theELUpdator->deltaLocalErrors(TSoS,propDir);
  if ( elWeights.empty() ) {
    //
    // create one dummy component
    //
    elWeights.push_back(1.);
    elDeltaPs.push_back(0.);
    elDeltaCovs.push_back(AlgebraicSymMatrix55());
  }
  //
  // combine the two multi-updates
  //
  std::vector<double>::const_iterator iMsWgt(msWeights.begin());
  std::vector<double>::const_iterator iMsDp(msDeltaPs.begin());
  std::vector<AlgebraicSymMatrix55>::const_iterator iMsDc(msDeltaCovs.begin());
  for ( ; iMsWgt<msWeights.end(); iMsWgt++,iMsDp++,iMsDc++ ) {

    std::vector<double>::const_iterator iElWgt(elWeights.begin());
    std::vector<double>::const_iterator iElDp(elDeltaPs.begin());
    std::vector<AlgebraicSymMatrix55>::const_iterator iElDc(elDeltaCovs.begin());
    for ( ; iElWgt<elWeights.end(); iElWgt++,iElDp++,iElDc++ ) {
      theWeights.push_back((*iMsWgt)*(*iElWgt));
      theDeltaPs.push_back((*iMsDp)+(*iElDp));
      theDeltaCovs.push_back((*iMsDc)+(*iElDc));
    }
  }

}

