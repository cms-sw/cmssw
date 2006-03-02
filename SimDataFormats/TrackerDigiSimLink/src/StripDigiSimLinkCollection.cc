#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLinkCollection.h"
#include <iostream>
 
void StripDigiSimLinkCollection::put(Range input, unsigned int detID) {
  // put in Digis of detID
 
  // store size of vector before put
  IndexRange inputRange;
 
  // put in StripDigiSimLinks from input
  bool first = true;
 
  // fill input in temporary vector for sorting
  std::vector<StripDigiSimLink> temporary;
  StripDigiSimLinkCollection::ContainerIterator sort_begin = input.first;
  StripDigiSimLinkCollection::ContainerIterator sort_end = input.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
    temporary.push_back(*sort_begin);
  }
  // std::sort(temporary.begin(),temporary.end());
 
  // iterators over input
  StripDigiSimLinkCollection::ContainerIterator begin = temporary.begin();
  StripDigiSimLinkCollection::ContainerIterator end = temporary.end();
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
 
const StripDigiSimLinkCollection::Range StripDigiSimLinkCollection::get(unsigned int detID) const {
  // get Digis of detID
 
  StripDigiSimLinkCollection::IndexRange returnIndexRange = map_[detID];
 
  StripDigiSimLinkCollection::Range returnRange;
  returnRange.first  = container_.begin()+returnIndexRange.first;
  returnRange.second = container_.begin()+returnIndexRange.second+1;
 
  return returnRange;
}
 
const std::vector<unsigned int> StripDigiSimLinkCollection::detIDs() const {
  // returns vector of detIDs in map
 
  StripDigiSimLinkCollection::RegistryIterator begin = map_.begin();
  StripDigiSimLinkCollection::RegistryIterator end   = map_.end();
 
  std::vector<unsigned int> output;
 
  for (; begin != end; ++begin) {
    output.push_back(begin->first);
  }
 
  return output;
 
}
