#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include "SimG4Core/Notification/interface/Singleton.h"

class TouchableToHistory;
class G4VPhysicalVolume;
class G4VTouchable;

class TrackerG4SimHitNumberingScheme 
{
public:
    TrackerG4SimHitNumberingScheme();
    int g4ToNumberingScheme(const G4VTouchable*);
    ~TrackerG4SimHitNumberingScheme();
private:
    TouchableToHistory * ts;
};

typedef Singleton<TrackerG4SimHitNumberingScheme> TkG4SimHitNumberingScheme;

#endif
