#ifndef SimG4Core_GenParticleInfo_H
#define SimG4Core_GenParticleInfo_H

#include "G4VUserPrimaryParticleInformation.hh"

class GenParticleInfo : public G4VUserPrimaryParticleInformation {
public:
  explicit GenParticleInfo(int id) : id_(id) {}
  ~GenParticleInfo() = default;
  int id() const { return id_; }
  void Print() const override {}

private:
  int id_;
};

#endif
