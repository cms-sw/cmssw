#ifndef TrajectoryStateTransform_H
#define TrajectoryStateTransform_H

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

class TrajectoryStateOnSurface;
class Surface;
class MagneticField;

class TrajectoryStateTransform {
public:

  PTrajectoryStateOnDet* persistentState( const TrajectoryStateOnSurface& ts,
					  unsigned int detid) const;

  TrajectoryStateOnSurface transientState( const PTrajectoryStateOnDet& ts,
					   const Surface* surface,
					   const MagneticField* field) const;

};

#endif
