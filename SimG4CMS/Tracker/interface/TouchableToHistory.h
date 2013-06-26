#ifndef SimG4CMS_Tracker_TouchableToHistory_H
#define SimG4CMS_Tracker_TouchableToHistory_H


#include "G4VTouchable.hh"

#include<vector>
#include<map>

class G4VTouchable;
class G4VPhysicalVolume;
class DDFilteredView;
class DDCompactView;
class GeometricDet;

class TouchableToHistory{
 public:
  // Nav_Story is G4
  // nav_type  is DDD
  typedef std::vector<int> nav_type;
  typedef std::vector<std::pair<int,std::string> > Nav_Story;
  typedef std::map <Nav_Story,nav_type> MapType;
  typedef std::map <Nav_Story,int> DirectMapType;
      TouchableToHistory(const DDCompactView&cpv, const GeometricDet& det): 
	 alreadySet(false), myCompactView(&cpv), myGeomDet(&det) {} 
  G4VPhysicalVolume& getTouchable(DDFilteredView&);
  Nav_Story getNavStory(DDFilteredView&);
  void buildAll();
  DDFilteredView& getFilteredView(const G4VTouchable&, DDFilteredView&);
  nav_type getNavType(const G4VTouchable&);
  Nav_Story touchableToNavStory(const G4VTouchable*);
  nav_type touchableToNavType(const G4VTouchable*);
  int touchableToInt(const G4VTouchable*);
 private:
  void dumpG4VPV(const G4VTouchable*);
  MapType myMap;
  DirectMapType myDirectMap;
  bool alreadySet;
  const DDCompactView* myCompactView;
  const GeometricDet* myGeomDet;
};

#endif


