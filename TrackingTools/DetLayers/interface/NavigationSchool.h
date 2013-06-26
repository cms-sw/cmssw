#ifndef DetLayers_NavigationSchool_H
#define DetLayers_NavigationSchool_H

#include <vector>

class NavigableLayer;
class NavigationSchool;
class DetLayer;

/** A base class for NavigationSchools.
 *  The links between layers are computed or loaded from 
 *  persistent store by a NavigationSchool.
 *  The result is a container of NavigableLayers.
 */


class NavigationSchool {
public:

  NavigationSchool() : theAllDetLayersInSystem(0){}

  virtual ~NavigationSchool() {}

  typedef std::vector<NavigableLayer*>   StateType;

  virtual StateType navigableLayers() const = 0;

  const std::vector<DetLayer*> & allLayersInSystem() const {return *theAllDetLayersInSystem;}

 protected:
  const std::vector<DetLayer*> * theAllDetLayersInSystem;
};

#endif // NavigationSchool_H
