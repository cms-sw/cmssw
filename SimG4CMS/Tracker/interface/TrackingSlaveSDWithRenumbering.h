/**
 * A TrackingSlaveSD with hooks for renumbering the trackid inside the hits
 */

#ifndef TrackingSlaveSDWithRenumbering_h
#define TrackingSlaveSDWithRenumbering_h

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
//#include "Utilities/Notification/interface/Observer.h"
//#include "Utilities/Notification/interface/LazyObserver.h"

class EndOfEvent;
class EventAction;

#include<string>

class TrackingSlaveSDWithRenumbering : public TrackingSlaveSD
//
//				       private Observer<const EndOfEvent *>,
//				       public LazyObserver<const EventAction*>
{
 public:
  TrackingSlaveSDWithRenumbering(std::string);
  void upDate(const EndOfEvent *);  
  void lazyUpDate(const  EventAction *);
  bool format();
 private:
  const EventAction* eventAction;
};

#endif // TrackingSlaveSD_h





