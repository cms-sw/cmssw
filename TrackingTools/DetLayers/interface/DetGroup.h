#ifndef TkDetLayers_DetGroup_h
#define TkDetLayers_DetGroup_h


#include <vector>
#include <utility>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class GeometricSearchDet;

class DetGroupElement {
 public:
  
  typedef GeometricSearchDet                                Det;
  typedef std::pair< const Det*, TrajectoryStateOnSurface>  DetWithState;

  DetGroupElement( const DetWithState& dws) :
    det_(dws.first), state_(dws.second) {}

  DetGroupElement( const Det* d, const TrajectoryStateOnSurface& s) :
    det_(d), state_(s) {}

  const Det* det() const {return det_;}
  const TrajectoryStateOnSurface& trajectoryState() const {return state_;}

private:

  const Det* det_;
  TrajectoryStateOnSurface state_;

};


class DetGroup : public std::vector< DetGroupElement> {
public:

  typedef std::vector< DetGroupElement>         Base;
  typedef DetGroupElement::DetWithState         DetWithState;

  DetGroup() {}
  DetGroup(int ind, int indSize) : index_(ind), indexSize_(indSize) {}

  DetGroup(const std::vector<DetWithState>& vec) {
    reserve( vec.size());
    for (std::vector<DetWithState>::const_iterator i=vec.begin(); i!=vec.end(); i++) {
      push_back(DetGroupElement(*i));
    }
  }

  int index() const {return index_;}

  int indexSize() const {return indexSize_;}

  void setIndexSize( int newSize) {indexSize_ = newSize;}

  void incrementIndex( int incr) {
    // for (iterator i=begin(); i!=end(); i++) i->incrementIndex(incr);
    index_ += incr;
    indexSize_ += incr;
  }

private:

  int index_;
  int indexSize_;

};

#endif
