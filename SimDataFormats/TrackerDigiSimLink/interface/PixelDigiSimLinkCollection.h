#ifndef TRACKINGOBJECTS_PIXELDIGISIMLINKCOLLECTION_H
#define TRACKINGOBJECTS_PIXELDIGISIMLINKCOLLECTION_H
 
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include <vector>
#include <map>
#include <utility>
 
class PixelDigiSimLinkCollection {
 
 public:
 
  typedef std::vector<PixelDigiSimLink>::const_iterator ContainerIterator;
  typedef std::pair<ContainerIterator, ContainerIterator> Range;
  typedef std::pair<unsigned int, unsigned int> IndexRange;
  typedef std::map<unsigned int, IndexRange> Registry;
  typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;
 
  PixelDigiSimLinkCollection() {}
   
  void put(Range input, unsigned int detID);
  const Range get(unsigned int detID) const;
  const std::vector<unsigned int> detIDs() const;
   
 private:
  mutable std::vector<PixelDigiSimLink> container_;
  mutable Registry map_;
 
};
 
#endif // TRACKINGOBJECTS_PIXELDIGISIMLINKCOLLECTION_H
