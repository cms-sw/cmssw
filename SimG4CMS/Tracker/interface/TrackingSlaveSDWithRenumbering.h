/**
 * A TrackingSlaveSD with hooks for renumbering the trackid inside the hits
 */

#ifndef TrackingSlaveSDWithRenumbering_h
#define TrackingSlaveSDWithRenumbering_h
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"


class EventAction;

#include<string>

class TrackingSlaveSDWithRenumbering : 
public TrackingSlaveSD, public Observer<const EndOfEvent *>
{
 public:
  TrackingSlaveSDWithRenumbering(std::string);
  void update(const EndOfEvent *);  
  void lazyUpDate(const  EventAction *);
  bool format();
 private:
  const EventAction* eventAction;
};

#endif // TrackingSlaveSD_h





