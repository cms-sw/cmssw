#ifndef SimG4Core_SensitiveDetectorCatalog_h
#define SimG4Core_SensitiveDetectorCatalog_h

#include <map>
#include <string>
#include <vector>

class SensitiveDetectorCatalog {
public:
  typedef std::map<std::string, std::vector<std::string>> MapType;
  void insert(const std::string &, const std::string &, const std::string &);
  const std::vector<std::string> &logicalNames(const std::string &readoutName) const;
  std::vector<std::string> logicalNamesFromClassName(const std::string &className) const;
  const std::vector<std::string> &readoutNames(const std::string &className) const;
  std::vector<std::string> readoutNames() const;
  std::string className(const std::string &readoutName) const;
  std::vector<std::string> classNames() const;

private:
  MapType theClassNameMap;
  MapType theROUNameMap;
};

#endif
