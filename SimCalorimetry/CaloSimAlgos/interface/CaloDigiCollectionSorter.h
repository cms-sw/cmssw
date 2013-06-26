#ifndef CaloDigiCollectionSorter_h
#define CaloDigiCollectionSorter_h

/** For test purposes, users might want to sort
    a collection of digis to find the highest 
    energies.  This class does that, and should
    work for all ECAL and HCAL digi types

    \Author Rick Wilkinson
*/

#include <vector>
#include <algorithm>
#include "DataFormats/Common/interface/SortedCollection.h"


class CaloDigiCollectionSorter {
public:
  CaloDigiCollectionSorter(int bin) : theMaxBin(bin) {}

  /// embedded class to be used as a sort predicate
  template<class T>
  class CaloDigiSortByMaxBin {
  public:
    CaloDigiSortByMaxBin(int bin) : theMaxBin(bin) {}

    bool operator()(const T & df1, const T & df2) const {
       // should work for HcalQIESamples & EcalMPGASamples
       // sort in reverse order, so highest bins come first
       return (df1[theMaxBin].raw() > df2[theMaxBin].raw());
    }

  private:
    int theMaxBin;
  };

  /// takes a digi collection and returns a vector of digis,
  /// sorted by the peak bin
  template<class T>
  std::vector<T> sortedVector(const edm::SortedCollection<T> & input) const {
     std::vector<T> result;
     result.reserve(input.size());
     for(unsigned int i = 0; i < input.size() ; ++i) 
     {
       result.push_back(input[i]);
     }
     // now sort
     std::sort(result.begin(), result.end(), CaloDigiSortByMaxBin<T>(theMaxBin));
     return result;
  }

private:
  int theMaxBin;
};

#endif

