#ifndef SimG4Core_GenParticleInfo_H
#define SimG4Core_GenParticleInfo_H

#include "G4VUserPrimaryParticleInformation.hh"

class GenParticleInfo : public G4VUserPrimaryParticleInformation 
{
public:
    explicit GenParticleInfo(int id) : id_(id) {}
    int id() const { return id_; }
    virtual void Print() const {}
private:
    int id_;
};

#endif
