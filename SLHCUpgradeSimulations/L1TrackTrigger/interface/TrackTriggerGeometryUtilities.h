#ifndef TRACK_TRIGGER_GEOMETRY_UTILITIES_a_H
#define TRACK_TRIGGER_GEOMETRY_UTILITIES_a_H

#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include <vector>

namespace cmsUpgrades{
namespace TrackTriggerGeometryUtilities{

GlobalPoint hitPosition(const GeomDetUnit* geom, const TrackTriggerHit& hit);

GlobalPoint averagePosition(const GeomDetUnit* geom, const std::vector<TrackTriggerHit>& hits);

}
}

#endif

