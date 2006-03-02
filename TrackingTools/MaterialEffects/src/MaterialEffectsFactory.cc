#include "TrackingTools/MaterialEffects/interface/MaterialEffectsFactory.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"

MaterialEffectsFactory::MaterialEffectsFactory()
{
  //  static SimpleConfigurable<string> updatorConf("CombinedMaterialEffectsUpdator","MaterialEffects:updatorName");
  //  theUpdatorName = updatorConf.value();
  theUpdatorName = "CombinedMaterialEffectsUpdator";//temporary hack
}

MaterialEffectsUpdator*
MaterialEffectsFactory::constructComponent()
{
  if ( theUpdatorName=="CombinedMaterialEffectsUpdator" )
    return new CombinedMaterialEffectsUpdator();
  if ( theUpdatorName=="EnergyLossUpdator" )
    return new EnergyLossUpdator();
  if ( theUpdatorName=="MultipleScatteringUpdator" )
    return new MultipleScatteringUpdator();
  return 0;
}

MaterialEffectsUpdator*
MaterialEffectsFactory::constructComponent(const float mass)
{
  if ( theUpdatorName=="CombinedMaterialEffectsUpdator" )
    return new CombinedMaterialEffectsUpdator(mass);
  if ( theUpdatorName=="EnergyLossUpdator" )
    return new EnergyLossUpdator(mass);
  if ( theUpdatorName=="MultipleScatteringUpdator" )
    return new MultipleScatteringUpdator(mass);
  return 0;
}
