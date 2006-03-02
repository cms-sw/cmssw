#ifndef TRACKINGOBJECTS_STRIPDIGISIMLINKCOLLECTION_H
#define TRACKINGOBJECTS_STRIPDIGISIMLINKCOLLECTION_H
 
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include <vector>
#include <map>
#include <utility>
 
class StripDigiSimLinkCollection {
 
 public:
 
  typedef std::vector<StripDigiSimLink>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;
 
  StripDigiSimLinkCollection() {}
   
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
   
 private:
  mutable std::vector<StripDigiSimLink> container_;
  mutable Registry map_;
 
};
 
#endif // TRACKINGOBJECTS_STRIPDIGISIMLINKCOLLECTION_H
