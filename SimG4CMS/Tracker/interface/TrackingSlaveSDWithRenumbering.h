/**
 * A TrackingSlaveSD with hooks for renumbering the trackid inside the hits
 */

#ifndef TrackingSlaveSDWithRenumbering_h
#define TrackingSlaveSDWithRenumbering_h
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

class SimTrackManager;

#include<string>

class TrackingSlaveSDWithRenumbering : 
public TrackingSlaveSD, public Observer<const EndOfEvent *>
{
 public:
  TrackingSlaveSDWithRenumbering(std::string, const SimTrackManager* );
  void update(const EndOfEvent *);  
  bool format();
 private:
  const SimTrackManager* m_trackManager;
};

#endif // TrackingSlaveSD_h





