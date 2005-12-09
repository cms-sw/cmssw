#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include <vector>

class TouchableToHistory;
class G4VPhysicalVolume;
class G4VTouchable;
class DDCompactView;
class GeometricDet;

class TrackerG4SimHitNumberingScheme 
{
public:
  typedef std::vector<int> nav_type;
  TrackerG4SimHitNumberingScheme(const DDCompactView&, const GeometricDet&);
  ~TrackerG4SimHitNumberingScheme(){clear();}
    
  void clear();

  unsigned int g4ToNumberingScheme(const G4VTouchable*);
  
private:
    TouchableToHistory * ts;
};



#endif
