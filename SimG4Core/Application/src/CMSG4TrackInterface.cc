#include "SimG4Core/Application/interface/CMSG4TrackInterface.h"

G4ThreadLocal CMSG4TrackInterface* CMSG4TrackInterface::interface_ = nullptr;

CMSG4TrackInterface* CMSG4TrackInterface::instance() {
  if (nullptr == interface_) {
    static G4ThreadLocalSingleton<CMSG4TrackInterface> inst;
    interface_ = inst.Instance();
  }
  return interface_;
}

CMSG4TrackInterface::CMSG4TrackInterface() {};

CMSG4TrackInterface::~CMSG4TrackInterface() {};
