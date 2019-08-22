#ifndef SimG4Core_SensitiveDetectorCatalog_h
#define SimG4Core_SensitiveDetectorCatalog_h

#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

class SensitiveDetectorCatalog {
public:
  using MapType = std::map<std::string, std::unordered_set<std::string>>;
  void insert(const std::string &, const std::string &, const std::string &);
  const std::vector<std::string_view> logicalNames(const std::string &readoutName) const;
  const std::vector<std::string_view> readoutNames(const std::string &className) const;
  std::vector<std::string_view> readoutNames() const;
  std::string_view className(const std::string &readoutName) const;
  void printMe() const;

private:
  std::vector<std::string_view> logicalNamesFromClassName(const std::string &className) const;
  std::vector<std::string_view> classNames() const;

  MapType theClassNameMap;
  MapType theROUNameMap;
};

#endif
