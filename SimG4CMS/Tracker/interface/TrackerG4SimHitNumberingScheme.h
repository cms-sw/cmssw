#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include <vector>

class TouchableToHistory;
class G4VPhysicalVolume;
class G4VTouchable;

class TrackerG4SimHitNumberingScheme 
{
public:
  typedef std::vector<int> nav_type;
  TrackerG4SimHitNumberingScheme();
  ~TrackerG4SimHitNumberingScheme(){clear();}
  
  static TrackerG4SimHitNumberingScheme& instance() {
    static TrackerG4SimHitNumberingScheme* theInstance = 0;
    if (!theInstance) theInstance = new TrackerG4SimHitNumberingScheme();
    return *theInstance;
  } 
  
  void clear();

  unsigned int g4ToNumberingScheme(const G4VTouchable*);
  
private:
    TouchableToHistory * ts;
};



#endif
