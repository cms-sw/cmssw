#ifndef RecoTracker_PixelVertexFinding_Cluster1DCleaner_h
#define RecoTracker_PixelVertexFinding_Cluster1DCleaner_h

#include "CommonTools/Clustering1D/interface/Cluster1D.h"

#include <cmath>

/*
 * given a vector<Cluster1D<T> >, erase Cluster1D further away than 
 * ZOffeSet from the average position, then 
 * recompute the vertex position. The ZOffeSet is taken as 
 * an aurgument
*/
namespace pixeltemp {
  template <class T>
  class Cluster1DCleaner {
  public:
    Cluster1DCleaner(const float zoffset, bool useErr) : theZOffSet(zoffset), theUseError(useErr) {
      theCleanedCluster1Ds.clear();
      theDiscardedCluster1Ds.clear();
    }
    // return the compatible clusters
    std::vector<Cluster1D<T> > clusters(const std::vector<Cluster1D<T> >&);
    /*
       return the vector of discarded Cluster1Ds
       it should be called after Cluster1DCleaner::clusters
       otherwise return an empty vector
    */
    std::vector<Cluster1D<T> > discardedCluster1Ds() const { return theDiscardedCluster1Ds; }

  private:
    void cleanCluster1Ds(const std::vector<Cluster1D<T> >&);
    float average(const std::vector<Cluster1D<T> >&);
    std::vector<Cluster1D<T> > theCleanedCluster1Ds;
    std::vector<Cluster1D<T> > theDiscardedCluster1Ds;
    float theZOffSet;
    bool theUseError;
  };

  /*
 *                                implementation
 */

  template <class T>
  std::vector<Cluster1D<T> > Cluster1DCleaner<T>::clusters(const std::vector<Cluster1D<T> >& clust) {
    cleanCluster1Ds(clust);
    return theCleanedCluster1Ds;
  }

  template <class T>
  void Cluster1DCleaner<T>::cleanCluster1Ds(const std::vector<Cluster1D<T> >& clust) {
    theCleanedCluster1Ds.clear();
    theDiscardedCluster1Ds.clear();
    if (clust.empty())
      return;
    float oldPos = average(clust);
    for (typename std::vector<Cluster1D<T> >::const_iterator ic = clust.begin(); ic != clust.end(); ic++) {
      float discr = theUseError ? fabs(((*ic).position().value() - oldPos) / (*ic).position().error())
                                : fabs(((*ic).position().value() - oldPos));
      if (discr < theZOffSet) {
        theCleanedCluster1Ds.push_back(*ic);
      } else {
        theDiscardedCluster1Ds.push_back(*ic);
      }
    }
    return;
  }

  //I could use the Cluster1DMerger...
  template <class T>
  float Cluster1DCleaner<T>::average(const std::vector<Cluster1D<T> >& clust) {
    //    float ave = clust.front().position().value();
    //    float err = clust.front().position().error();
    //    for( typename std::vector < Cluster1D<T> >::const_iterator ic=(clust.begin())+1;
    //         ic != clust.end(); ic++)
    //    {
    //        float oldave = ave;
    //        float olderr = err;
    //        ave = ( oldave/olderr/olderr +
    //                ic->position().value()/ic->position().error()/ic->position().error()) /
    //              (1./olderr/olderr + 1./ic->position().error()/ic->position().error());
    //        err = sqrt(olderr*olderr + ic->position().error()*ic->position().error());
    //    }
    float sumUp = 0;
    float sumDown = 0;
    for (typename std::vector<Cluster1D<T> >::const_iterator ic = (clust.begin()) + 1; ic != clust.end(); ic++) {
      float err2 = ic->position().error();
      err2 *= err2;
      if (err2 != 0) {
        sumUp += ic->position().value() / err2;  // error-weighted average of Z at IP
        sumDown += 1 / err2;
      }
    }
    return (sumDown > 0) ? sumUp / sumDown : 0;
  }
}  // namespace pixeltemp
#endif
