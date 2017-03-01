#ifndef SimG4Core_ComponentFactoryByName_H 
#define SimG4Core_ComponentFactoryByName_H 

#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <map>

template <class B>
class ComponentFactoryByName
{
public:
  static B * getBuilder(const std::string & name)
    {
      if (myMap().size() == 0) {
	throw cms::Exception("SimG4CoreNotification", " ComponentFactoryByName::getBuilder: No Builder registered to the Factory.");
      }
      if (myMap().find(name) == myMap().end()) {
	throw cms::Exception("SimG4CoreNotification", " ComponentFactoryByName::getBuilder: The Component "+name+" is not registered to the Factory.");
      }
      return (myMap()[name]);
    }
  static void setBuilder(B * in , const std::string & name)
    {
      if (name.empty()) {
        throw cms::Exception("SimG4CoreNotification", " ComponentFactoryByName::setBuilder: The registration of Components without name is not allowed.");
      }
      myMap()[name] = in;
    }
    typedef std::map<std::string,B *> BuilderMapType;
protected:
    static BuilderMapType & myMap()
    {
	static BuilderMapType me_;
	return me_;
    }
};

#endif



