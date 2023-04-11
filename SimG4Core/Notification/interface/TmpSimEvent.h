#ifndef SimG4Core_TmpSimEvent_H
#define SimG4Core_TmpSimEvent_H

#include "SimG4Core/Notification/interface/TmpSimTrack.h"
#include "SimG4Core/Notification/interface/TmpSimVertex.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "HepMC/GenEvent.h"

#include <vector>

class TmpSimEvent {
public:
  TmpSimEvent();
  virtual ~TmpSimEvent();
  void load(edm::SimTrackContainer& c) const;
  void load(edm::SimVertexContainer& c) const;
  unsigned int nTracks() const { return g4tracks_.size(); }
  unsigned int nVertices() const { return g4vertices_.size(); }
  unsigned int nGenParts() const { return hepMCEvent_->particles_size(); }
  void hepEvent(const HepMC::GenEvent* r) { hepMCEvent_ = r; }
  const HepMC::GenEvent* hepEvent() const { return hepMCEvent_; }
  void weight(float w) { weight_ = w; }
  float weight() const { return weight_; }
  void collisionPoint(const math::XYZTLorentzVectorD& v) { collisionPoint_ = v; }
  const math::XYZTLorentzVectorD& collisionPoint() const { return collisionPoint_; }
  void nparam(int n) { nparam_ = n; }
  const int nparam() const { return nparam_; }
  void param(const std::vector<float>& p) { param_ = p; }
  const std::vector<float>& param() const { return param_; }
  void add(TmpSimTrack* t) { g4tracks_.push_back(t); }
  void add(TmpSimVertex* v) { g4vertices_.push_back(v); }
  void clear();

private:
  const HepMC::GenEvent* hepMCEvent_{nullptr};
  float weight_{0.f};
  math::XYZTLorentzVectorD collisionPoint_{math::XYZTLorentzVectorD(0., 0., 0., 0.)};
  int nparam_{0};
  std::vector<float> param_;
  std::vector<TmpSimTrack*> g4tracks_;
  std::vector<TmpSimVertex*> g4vertices_;
};

#endif
