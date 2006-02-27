#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"
#include <iostream>
#include <vector> 
void PixelDigiSimLinkCollection::put(Range input, unsigned int detID) {
  // put in Digis of detID
 
  // store size of vector before put
  IndexRange inputRange;
 
  // put in PixelDigiSimLinks from input
  bool first = true;
 
  // fill input in temporary vector for sorting
  std::vector<PixelDigiSimLink> temporary;
  PixelDigiSimLinkCollection::ContainerIterator sort_begin = input.first;
  PixelDigiSimLinkCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  // std::sort(temporary.begin(),temporary.end());
 
  // iterators over input
  PixelDigiSimLinkCollection::ContainerIterator begin = temporary.begin();
  PixelDigiSimLinkCollection::ContainerIterator end = temporary.end();
  for ( ;begin != end; ++begin ) {
    container_.push_back(*begin);
    if ( first ) {
      inputRange.first = container_.size()-1;
      first = false;
    }
  }
  inputRange.second = container_.size()-1;
   
  // fill map
  map_[detID] = inputRange;
 
}
 
const PixelDigiSimLinkCollection::Range PixelDigiSimLinkCollection::get(unsigned int detID) const {
  // get Digis of detID
 
  PixelDigiSimLinkCollection::IndexRange returnIndexRange = map_[detID];
 
  PixelDigiSimLinkCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;
 
  return returnRange;
}
 
const std::vector<unsigned int> PixelDigiSimLinkCollection::detIDs() const {
  // returns vector of detIDs in map
 
  PixelDigiSimLinkCollection::RegistryIterator begin = map_.begin();
  PixelDigiSimLinkCollection::RegistryIterator end   = map_.end();
 
  std::vector<unsigned int> output;
 
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }
 
  return output;
 
}
