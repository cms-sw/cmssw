#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include <vector>
#include <map>
#include <string>

class G4VTouchable;
class G4VPhysicalVolume;
class DDFilteredView;
class DDCompactView;
class GeometricDet;

class TrackerG4SimHitNumberingScheme {
public:
  // Nav_Story is G4
  // nav_type  is DDD
  typedef std::vector<int> Nav_type;
  typedef std::vector<std::pair<int, std::string> > Nav_Story;
  typedef std::map<Nav_Story, Nav_type> MapType;
  typedef std::map<Nav_Story, unsigned int> DirectMapType;

  TrackerG4SimHitNumberingScheme(const DDCompactView&, const GeometricDet&);
  ~TrackerG4SimHitNumberingScheme();

  unsigned int g4ToNumberingScheme(const G4VTouchable*);

  const G4VPhysicalVolume& getTouchable(DDFilteredView&);
  const DDFilteredView& getFilteredView(const G4VTouchable&, DDFilteredView&);

private:
  Nav_type& getNavType(const G4VTouchable&);
  Nav_type& touchableToNavType(const G4VTouchable*);
  void getNavStory(DDFilteredView&, Nav_Story&);
  void touchToNavStory(const G4VTouchable*, Nav_Story&);
  void dumpG4VPV(const G4VTouchable*);

  void buildAll();

  MapType myMap;
  DirectMapType myDirectMap;
  bool alreadySet;
  const DDCompactView* myCompactView;
  const GeometricDet* myGeomDet;
};

#endif
