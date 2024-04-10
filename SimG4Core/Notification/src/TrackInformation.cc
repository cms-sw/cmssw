#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>

G4ThreadLocal G4Allocator<TrackInformation>* fpTrackInformationAllocator = nullptr;

void TrackInformation::setCrossedBoundary(const G4Track* track) {
  const double invcm = 1.0 / CLHEP::cm;
  const double invgev = 1.0 / CLHEP::GeV;
  const double invsec = 1.0 / CLHEP::second;
  crossedBoundary_ = true;
  const G4ThreeVector& v = track->GetPosition();
  positionAtBoundary_ =
      math::XYZTLorentzVectorF(v.x() * invcm, v.y() * invcm, v.z() * invcm, track->GetGlobalTime() * invsec);
  const G4ThreeVector& p = track->GetMomentum();
  momentumAtBoundary_ =
      math::XYZTLorentzVectorF(p.x() * invgev, p.y() * invgev, p.z() * invgev, track->GetTotalEnergy() * invgev);
}

void TrackInformation::Print() const {
  LogDebug("TrackInformation") << " TrackInformation : storeTrack = " << storeTrack_ << "\n"
                               << "                    hasHits = " << hasHits_ << "\n"
                               << "                    isPrimary = " << isPrimary_ << "\n"
                               << "                    isGeneratedSecondary = " << isGeneratedSecondary_ << "\n"
                               << "                    mcTruthID = " << mcTruthID_ << "\n"
                               << "                    isInHistory = " << isInHistory_ << "\n"
                               << "                    idOnCaloSurface = " << getIDonCaloSurface() << "\n"
                               << "                    caloIDChecked = " << caloIDChecked() << "\n"
                               << "                    idCaloVolume = " << idCaloVolume_ << "\n"
                               << "                    idLastVolume = " << idLastVolume_ << "\n"
                               << "                    isFromTtoBTL = " << isFromTtoBTL() << "\n"
                               << "                    isFromBTLtoT = " << isFromBTLtoT() << "\n"
                               << "                    isBTLlooper = " << isBTLlooper() << "\n"
                               << "                    isInTrkFromBackscattering = " << isInTrkFromBackscattering()
                               << "\n"
                               << "                    isExtSecondary = " << isExtSecondary();
}
