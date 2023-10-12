#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include <vector>
#include <map>
#include <string>
#include <G4VTouchable.hh>

class G4VPhysicalVolume;
class GeometricDet;

class TrackerG4SimHitNumberingScheme {
public:
  // Nav_Story is G4
  using Nav_Story = std::vector<std::pair<int, std::string> >;
  using DirectMapType = std::map<Nav_Story, unsigned int>;

  explicit TrackerG4SimHitNumberingScheme(const GeometricDet&);

  unsigned int g4ToNumberingScheme(const G4VTouchable*);

private:
  void touchToNavStory(const G4VTouchable*, Nav_Story&);
  void dumpG4VPV(const G4VTouchable*);

  void buildAll();

  DirectMapType directMap_;
  bool alreadySet_;
  const GeometricDet* geomDet_;
};

#endif
