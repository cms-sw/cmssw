#ifndef SimG4Core_SDCatalog_h
#define SimG4Core_SDCatalog_h

#include "SimG4Core/Notification/interface/Singleton.h"
 
#include <vector>
#include <map>
#include <string>

class SDCatalog 
{
public:
    typedef std::map<std::string,std::vector<std::string> > MapType;
    void insert(std::string &, std::string&, std::string&);
    std::vector<std::string> logicalNames(std::string & readoutName);
    std::vector<std::string> logicalNamesFromClassName(std::string & className);
    std::vector<std::string> readoutNames(std::string & className);
    std::vector<std::string> readoutNames();
    std::string className(std::string & readoutName);
    std::vector<std::string> classNames();
private:
    MapType theClassNameMap;
    MapType theROUNameMap;
};

typedef Singleton<SDCatalog> SensitiveDetectorCatalog;
 
#endif
