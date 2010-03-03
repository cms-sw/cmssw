#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerGeometryUtilities.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

GlobalPoint cmsUpgrades::TrackTriggerGeometryUtilities::hitPosition(const GeomDetUnit* geom, const TrackTriggerHit& hit)
{
	MeasurementPoint mp( hit.row() + 0.5, hit.column() + 0.5 ); // Add 0.5 to get the center of the pixel.
	return geom->surface().toGlobal( geom->topology().localPosition( mp ) ) ;
}

GlobalPoint cmsUpgrades::TrackTriggerGeometryUtilities::averagePosition(const GeomDetUnit* geom, const std::vector<TrackTriggerHit>& hits)
{
	double averageX = 0;
	double averageY = 0;
	double averageZ = 0;
	for ( std::vector<TrackTriggerHit>::const_iterator hits_itr = hits.begin(); hits_itr != hits.end(); hits_itr++ )
	{
		GlobalPoint thisHitPosition = hitPosition( geom, *hits_itr );
		averageX += thisHitPosition.x();
		averageY += thisHitPosition.y();
		averageZ += thisHitPosition.z();
	}
	averageX /= hits.size();
	averageY /= hits.size();
	averageZ /= hits.size();
	return GlobalPoint(averageX, averageY, averageZ);
}

