#ifndef DetLayers_DetGroup_h
#define DetLayers_DetGroup_h

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include <vector>
#include <utility>
#include <algorithm>

class DetGroupElement {
public:
  typedef std::pair<const GeomDet*, TrajectoryStateOnSurface> DetWithState;
  typedef GeomDet Det;

  DetGroupElement(const DetWithState& dws) : det_(dws.first), state_(dws.second) {}

  DetGroupElement(const Det* d, const TrajectoryStateOnSurface& s) : det_(d), state_(s) {}

  DetGroupElement(DetGroupElement const& rhs) : det_(rhs.det_), state_(rhs.state_) {}
  DetGroupElement(DetGroupElement&& rhs) noexcept : det_(rhs.det_), state_(std::move(rhs.state_)) {}
  DetGroupElement& operator=(DetGroupElement const& rhs) {
    det_ = rhs.det_;
    state_ = rhs.state_;
    return *this;
  }
  DetGroupElement& operator=(DetGroupElement&& rhs) noexcept {
    det_ = rhs.det_;
    state_ = std::move(rhs.state_);
    return *this;
  }
  DetGroupElement(const Det* d, TrajectoryStateOnSurface&& s) noexcept : det_(d), state_(std::move(s)) {}

  const Det* det() const { return det_; }
  const TrajectoryStateOnSurface& trajectoryState() const { return state_; }

private:
  const Det* det_;
  TrajectoryStateOnSurface state_;
};

class DetGroup : public std::vector<DetGroupElement> {
public:
  typedef std::vector<DetGroupElement> Base;
  typedef DetGroupElement::DetWithState DetWithState;

  DetGroup() : index_(0), indexSize_(0) {}
  DetGroup(DetGroup const& rhs) : Base(rhs), index_(rhs.index_), indexSize_(rhs.indexSize_) {}
  DetGroup(DetGroup&& rhs) noexcept : Base(std::forward<Base>(rhs)), index_(rhs.index_), indexSize_(rhs.indexSize_) {}
  DetGroup& operator=(DetGroup const& rhs) {
    Base::operator=(rhs);
    index_ = rhs.index_;
    indexSize_ = rhs.indexSize_;
    return *this;
  }
  DetGroup& operator=(DetGroup&& rhs) noexcept {
    Base::operator=(std::forward<Base>(rhs));
    index_ = rhs.index_;
    indexSize_ = rhs.indexSize_;
    return *this;
  }

  DetGroup(int ind, int indSize) : index_(ind), indexSize_(indSize) {}

  DetGroup(const std::vector<DetWithState>& vec) {
    reserve(vec.size());
    for (std::vector<DetWithState>::const_iterator i = vec.begin(); i != vec.end(); i++) {
      push_back(DetGroupElement(*i));
    }
  }

  int index() const { return index_; }

  int indexSize() const { return indexSize_; }

  void setIndexSize(int newSize) { indexSize_ = newSize; }

  void incrementIndex(int incr) {
    // for (iterator i=begin(); i!=end(); i++) i->incrementIndex(incr);
    index_ += incr;
    indexSize_ += incr;
  }

private:
  int index_;
  int indexSize_;
};

#endif
