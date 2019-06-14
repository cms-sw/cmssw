#ifndef SimG4Core_SensitiveDetectorCatalog_h
#define SimG4Core_SensitiveDetectorCatalog_h

#include <map>
#include <string>
#include <vector>

class SensitiveDetectorCatalog {
public:
  using MapType = std::map<std::string, std::vector<std::string>>;
  void insert(const std::string &, const std::string &, const std::string &);
  const std::vector<std::string> &logicalNames(const std::string &readoutName) const;
  const std::vector<std::string> &readoutNames(const std::string &className) const;
  std::vector<std::string> readoutNames() const;
  std::string className(const std::string &readoutName) const;

private:
  std::vector<std::string> logicalNamesFromClassName(const std::string &className) const;
  std::vector<std::string> classNames() const;
  
  MapType theClassNameMap;
  MapType theROUNameMap;
};

#endif
