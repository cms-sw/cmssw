#ifndef SiTrackerMRHTools_MeasurementByLayerGrouper_H
#define SiTrackerMRHTools_MeasurementByLayerGrouper_H

class DetLayer;
class TrajectoryMeasurement;
class GeometricSearchTracker;

#include <vector>
#include <map>

//groups the TrajectoryMeasurements on a layer by layer basis

class MeasurementByLayerGrouper {

private:

  typedef TrajectoryMeasurement TM;
  const GeometricSearchTracker* theGeomSearch; 

  const DetLayer* getDetLayer(const TM& tm) const;

public:

  explicit MeasurementByLayerGrouper(const GeometricSearchTracker* search = nullptr):theGeomSearch(search){};

  std::vector<std::pair<const DetLayer*, std::vector<TM> > > operator()(const std::vector<TM>&) const;


//to be ported later if needed
/*
  vector<TM> 
  operator()(const vector<pair<const DetLayer*, vector<TM> > >&) const;

  vector<pair<const DetLayer*, map<int, vector<TrajectoryMeasurement> > > > 
  operator()(const map<int, vector<TM> >&) const;
*/

};
#endif
