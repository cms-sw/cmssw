#include <map>
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "DetIdInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/VectorUtil.h"
#include <algorithm>

///////////////////////////

std::string TrackDetMatchInfo::dumpGeometry(const DetId& id) {
  if (!caloGeometry || !caloGeometry->getSubdetectorGeometry(id) ||
      !caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)) {
    throw cms::Exception("FatalError") << "Failed to access geometry for DetId: " << id.rawId();
  }
  std::ostringstream oss;

  const CaloCellGeometry::CornersVec& points = caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)->getCorners();
  for (const auto& point : points)
    oss << "(" << point.z() << ", " << point.perp() << ", " << point.eta() << ", " << point.phi() << "), \t";
  return oss.str();
}

GlobalPoint TrackDetMatchInfo::getPosition(const DetId& id) {
  // this part might be slow
  if (!caloGeometry || !caloGeometry->getSubdetectorGeometry(id) ||
      !caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)) {
    throw cms::Exception("FatalError") << "Failed to access geometry for DetId: " << id.rawId();
    return GlobalPoint(0, 0, 0);
  }
  return caloGeometry->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
}

double TrackDetMatchInfo::crossedEnergy(EnergyType type) {
  double energy(0);
  switch (type) {
    case EcalRecHits: {
      for (auto crossedEcalRecHit : crossedEcalRecHits)
        energy += crossedEcalRecHit->energy();
    } break;
    case HcalRecHits: {
      for (auto crossedHcalRecHit : crossedHcalRecHits)
        energy += crossedHcalRecHit->energy();
    } break;
    case HORecHits: {
      for (auto crossedHORecHit : crossedHORecHits)
        energy += crossedHORecHit->energy();
    } break;
    case TowerTotal: {
      for (auto crossedTower : crossedTowers)
        energy += crossedTower->energy();
    } break;
    case TowerEcal: {
      for (auto crossedTower : crossedTowers)
        energy += crossedTower->emEnergy();
    } break;
    case TowerHcal: {
      for (auto crossedTower : crossedTowers)
        energy += crossedTower->hadEnergy();
    } break;
    case TowerHO: {
      for (auto crossedTower : crossedTowers)
        energy += crossedTower->outerEnergy();
    } break;
    default:
      throw cms::Exception("FatalError") << "Unknown calo energy type: " << type;
  }
  return energy;
}

bool TrackDetMatchInfo::insideCone(const DetId& id, const double dR) {
  GlobalPoint idPosition = getPosition(id);
  if (idPosition.mag() < 0.01)
    return false;

  math::XYZVector idPositionRoot(idPosition.x(), idPosition.y(), idPosition.z());
  math::XYZVector trackP3(stateAtIP.momentum().x(), stateAtIP.momentum().y(), stateAtIP.momentum().z());
  return ROOT::Math::VectorUtil::DeltaR(trackP3, idPositionRoot) < dR;
}

double TrackDetMatchInfo::coneEnergy(double dR, EnergyType type) {
  double energy(0);
  switch (type) {
    case EcalRecHits: {
      for (auto ecalRecHit : ecalRecHits)
        if (insideCone(ecalRecHit->detid(), dR))
          energy += ecalRecHit->energy();
    } break;
    case HcalRecHits: {
      for (auto hcalRecHit : hcalRecHits)
        if (insideCone(hcalRecHit->detid(), dR))
          energy += hcalRecHit->energy();
    } break;
    case HORecHits: {
      for (auto hoRecHit : hoRecHits)
        if (insideCone(hoRecHit->detid(), dR))
          energy += hoRecHit->energy();
    } break;
    case TowerTotal: {
      for (auto crossedTower : crossedTowers)
        if (insideCone(crossedTower->id(), dR))
          energy += crossedTower->energy();
    } break;
    case TowerEcal: {
      for (auto crossedTower : crossedTowers)
        if (insideCone(crossedTower->id(), dR))
          energy += crossedTower->emEnergy();
    } break;
    case TowerHcal: {
      for (auto crossedTower : crossedTowers)
        if (insideCone(crossedTower->id(), dR))
          energy += crossedTower->hadEnergy();
    } break;
    case TowerHO: {
      for (auto crossedTower : crossedTowers)
        if (insideCone(crossedTower->id(), dR))
          energy += crossedTower->outerEnergy();
    } break;
    default:
      throw cms::Exception("FatalError") << "Unknown calo energy type: " << type;
  }
  return energy;
}

//////////////////////////////////////////////////

double TrackDetMatchInfo::nXnEnergy(const DetId& id, EnergyType type, int gridSize) {
  double energy(0);
  if (id.rawId() == 0)
    return 0.;
  switch (type) {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO: {
      if (id.det() != DetId::Calo) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected CaloTower, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      CaloTowerDetId centerId(id);
      for (auto tower : towers) {
        CaloTowerDetId neighborId(tower->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          switch (type) {
            case TowerTotal:
              energy += tower->energy();
              break;
            case TowerEcal:
              energy += tower->emEnergy();
              break;
            case TowerHcal:
              energy += tower->hadEnergy();
              break;
            case TowerHO:
              energy += tower->outerEnergy();
              break;
            default:
              throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
          }
        }
      }
    } break;
    case EcalRecHits: {
      if (id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      // Since the ECAL granularity is small and the gap between EE and EB is significant,
      // energy is computed only within the system that contains the central element
      if (id.subdetId() == EcalBarrel) {
        EBDetId centerId(id);
        for (auto ecalRecHit : ecalRecHits) {
          if (ecalRecHit->id().subdetId() != EcalBarrel)
            continue;
          EBDetId neighborId(ecalRecHit->id());
          int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                         (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
          int dPhi = abs(centerId.iphi() - neighborId.iphi());
          if (abs(360 - dPhi) < dPhi)
            dPhi = 360 - dPhi;
          if (dEta <= gridSize && dPhi <= gridSize) {
            energy += ecalRecHit->energy();
          }
        }
      } else {
        // Endcap
        EEDetId centerId(id);
        for (auto ecalRecHit : ecalRecHits) {
          if (ecalRecHit->id().subdetId() != EcalEndcap)
            continue;
          EEDetId neighborId(ecalRecHit->id());
          if (centerId.zside() == neighborId.zside() && abs(centerId.ix() - neighborId.ix()) <= gridSize &&
              abs(centerId.iy() - neighborId.iy()) <= gridSize) {
            energy += ecalRecHit->energy();
          }
        }
      }
    } break;
    case HcalRecHits: {
      if (id.det() != DetId::Hcal || (id.subdetId() != HcalBarrel && id.subdetId() != HcalEndcap)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected HE or HB, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      HcalDetId centerId(id);
      for (auto hcalRecHit : hcalRecHits) {
        HcalDetId neighborId(hcalRecHit->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          energy += hcalRecHit->energy();
        }
      }
    } break;
    case HORecHits: {
      if (id.det() != DetId::Hcal || (id.subdetId() != HcalOuter)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected HO, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      HcalDetId centerId(id);
      for (auto hoRecHit : hoRecHits) {
        HcalDetId neighborId(hoRecHit->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          energy += hoRecHit->energy();
        }
      }
    } break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
  }
  return energy;
}

double TrackDetMatchInfo::nXnEnergy(EnergyType type, int gridSize) {
  switch (type) {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
      if (crossedTowerIds.empty())
        return 0;
      return nXnEnergy(crossedTowerIds.front(), type, gridSize);
      break;
    case EcalRecHits:
      if (crossedEcalIds.empty())
        return 0;
      return nXnEnergy(crossedEcalIds.front(), type, gridSize);
      break;
    case HcalRecHits:
      if (crossedHcalIds.empty())
        return 0;
      return nXnEnergy(crossedHcalIds.front(), type, gridSize);
      break;
    case HORecHits:
      if (crossedHOIds.empty())
        return 0;
      return nXnEnergy(crossedHOIds.front(), type, gridSize);
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
  }
  return -999;
}

TrackDetMatchInfo::TrackDetMatchInfo()
    : trkGlobPosAtEcal(0, 0, 0),
      trkGlobPosAtHcal(0, 0, 0),
      trkGlobPosAtHO(0, 0, 0),
      trkMomAtEcal(0, 0, 0),
      trkMomAtHcal(0, 0, 0),
      trkMomAtHO(0, 0, 0),
      isGoodEcal(false),
      isGoodHcal(false),
      isGoodCalo(false),
      isGoodHO(false),
      isGoodMuon(false),
      simTrack(nullptr),
      ecalTrueEnergy(-999),
      hcalTrueEnergy(-999) {}

DetId TrackDetMatchInfo::findMaxDeposition(EnergyType type) {
  DetId id;
  float maxEnergy = -9999;
  switch (type) {
    case EcalRecHits: {
      for (auto ecalRecHit : ecalRecHits)
        if (ecalRecHit->energy() > maxEnergy) {
          maxEnergy = ecalRecHit->energy();
          id = ecalRecHit->detid();
        }
    } break;
    case HcalRecHits: {
      for (auto hcalRecHit : hcalRecHits)
        if (hcalRecHit->energy() > maxEnergy) {
          maxEnergy = hcalRecHit->energy();
          id = hcalRecHit->detid();
        }
    } break;
    case HORecHits: {
      for (auto hoRecHit : hoRecHits)
        if (hoRecHit->energy() > maxEnergy) {
          maxEnergy = hoRecHit->energy();
          id = hoRecHit->detid();
        }
    } break;
    case TowerTotal:
    case TowerEcal:
    case TowerHcal:
    case TowerHO: {
      for (auto tower : towers) {
        double energy = 0;
        switch (type) {
          case TowerTotal:
            energy = tower->energy();
            break;
          case TowerEcal:
            energy = tower->emEnergy();
            break;
          case TowerHcal:
            energy = tower->hadEnergy();
            break;
          case TowerHO:
            energy = tower->energy();
            break;
          default:
            throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
        }
        if (energy > maxEnergy) {
          maxEnergy = energy;
          id = tower->id();
        }
      }
    } break;
    default:
      throw cms::Exception("FatalError")
          << "Maximal energy deposition: unkown or not implemented energy type requested, type:" << type;
  }
  return id;
}

DetId TrackDetMatchInfo::findMaxDeposition(const DetId& id, EnergyType type, int gridSize) {
  double energy_max(0);
  DetId id_max;
  if (id.rawId() == 0)
    return id_max;
  switch (type) {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO: {
      if (id.det() != DetId::Calo) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected CaloTower, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      CaloTowerDetId centerId(id);
      for (auto tower : towers) {
        CaloTowerDetId neighborId(tower->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          switch (type) {
            case TowerTotal:
              if (energy_max < tower->energy()) {
                energy_max = tower->energy();
                id_max = tower->id();
              }
              break;
            case TowerEcal:
              if (energy_max < tower->emEnergy()) {
                energy_max = tower->emEnergy();
                id_max = tower->id();
              }
              break;
            case TowerHcal:
              if (energy_max < tower->hadEnergy()) {
                energy_max = tower->hadEnergy();
                id_max = tower->id();
              }
              break;
            case TowerHO:
              if (energy_max < tower->outerEnergy()) {
                energy_max = tower->outerEnergy();
                id_max = tower->id();
              }
              break;
            default:
              throw cms::Exception("FatalError") << "Unknown calo tower energy type: " << type;
          }
        }
      }
    } break;
    case EcalRecHits: {
      if (id.det() != DetId::Ecal || (id.subdetId() != EcalBarrel && id.subdetId() != EcalEndcap)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected EcalBarrel or EcalEndcap, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      // Since the ECAL granularity is small and the gap between EE and EB is significant,
      // energy is computed only within the system that contains the central element
      if (id.subdetId() == EcalBarrel) {
        EBDetId centerId(id);
        for (auto ecalRecHit : ecalRecHits) {
          if (ecalRecHit->id().subdetId() != EcalBarrel)
            continue;
          EBDetId neighborId(ecalRecHit->id());
          int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                         (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
          int dPhi = abs(centerId.iphi() - neighborId.iphi());
          if (abs(360 - dPhi) < dPhi)
            dPhi = 360 - dPhi;
          if (dEta <= gridSize && dPhi <= gridSize) {
            if (energy_max < ecalRecHit->energy()) {
              energy_max = ecalRecHit->energy();
              id_max = ecalRecHit->id();
            }
          }
        }
      } else {
        // Endcap
        EEDetId centerId(id);
        for (auto ecalRecHit : ecalRecHits) {
          if (ecalRecHit->id().subdetId() != EcalEndcap)
            continue;
          EEDetId neighborId(ecalRecHit->id());
          if (centerId.zside() == neighborId.zside() && abs(centerId.ix() - neighborId.ix()) <= gridSize &&
              abs(centerId.iy() - neighborId.iy()) <= gridSize) {
            if (energy_max < ecalRecHit->energy()) {
              energy_max = ecalRecHit->energy();
              id_max = ecalRecHit->id();
            }
          }
        }
      }
    } break;
    case HcalRecHits: {
      if (id.det() != DetId::Hcal || (id.subdetId() != HcalBarrel && id.subdetId() != HcalEndcap)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected HE or HB, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      HcalDetId centerId(id);
      for (auto hcalRecHit : hcalRecHits) {
        HcalDetId neighborId(hcalRecHit->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          if (energy_max < hcalRecHit->energy()) {
            energy_max = hcalRecHit->energy();
            id_max = hcalRecHit->id();
          }
        }
      }
    } break;
    case HORecHits: {
      if (id.det() != DetId::Hcal || (id.subdetId() != HcalOuter)) {
        throw cms::Exception("FatalError") << "Wrong DetId. Expected HO, but found:\n"
                                           << DetIdInfo::info(id, nullptr) << "\n";
      }
      HcalDetId centerId(id);
      for (auto hoRecHit : hoRecHits) {
        HcalDetId neighborId(hoRecHit->id());
        int dEta = abs((centerId.ieta() < 0 ? centerId.ieta() + 1 : centerId.ieta()) -
                       (neighborId.ieta() < 0 ? neighborId.ieta() + 1 : neighborId.ieta()));
        int dPhi = abs(centerId.iphi() - neighborId.iphi());
        if (abs(72 - dPhi) < dPhi)
          dPhi = 72 - dPhi;
        if (dEta <= gridSize && dPhi <= gridSize) {
          if (energy_max < hoRecHit->energy()) {
            energy_max = hoRecHit->energy();
            id_max = hoRecHit->id();
          }
        }
      }
    } break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
  }
  return id_max;
}

DetId TrackDetMatchInfo::findMaxDeposition(EnergyType type, int gridSize) {
  DetId id_max;
  switch (type) {
    case TowerTotal:
    case TowerHcal:
    case TowerEcal:
    case TowerHO:
      if (crossedTowerIds.empty())
        return id_max;
      return findMaxDeposition(crossedTowerIds.front(), type, gridSize);
      break;
    case EcalRecHits:
      if (crossedEcalIds.empty())
        return id_max;
      return findMaxDeposition(crossedEcalIds.front(), type, gridSize);
      break;
    case HcalRecHits:
      if (crossedHcalIds.empty())
        return id_max;
      return findMaxDeposition(crossedHcalIds.front(), type, gridSize);
      break;
    case HORecHits:
      if (crossedHOIds.empty())
        return id_max;
      return findMaxDeposition(crossedHOIds.front(), type, gridSize);
      break;
    default:
      throw cms::Exception("FatalError") << "Unkown or not implemented energy type requested, type:" << type;
  }
  return id_max;
}

////////////////////////////////////////////////////////////////////////
// Obsolete
//

double TrackDetMatchInfo::ecalConeEnergy() { return coneEnergy(999, EcalRecHits); }

double TrackDetMatchInfo::hcalConeEnergy() { return coneEnergy(999, HcalRecHits); }

double TrackDetMatchInfo::hoConeEnergy() { return coneEnergy(999, HcalRecHits); }

double TrackDetMatchInfo::ecalCrossedEnergy() { return crossedEnergy(EcalRecHits); }

double TrackDetMatchInfo::hcalCrossedEnergy() { return crossedEnergy(HcalRecHits); }

double TrackDetMatchInfo::hoCrossedEnergy() { return crossedEnergy(HORecHits); }

int TrackDetMatchInfo::numberOfSegments() const {
  int numSegments = 0;
  for (const auto& chamber : chambers)
    numSegments += chamber.segments.size();
  return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station) const {
  int numSegments = 0;
  for (const auto& chamber : chambers)
    if (chamber.station() == station)
      numSegments += chamber.segments.size();
  return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInStation(int station, int detector) const {
  int numSegments = 0;
  for (const auto& chamber : chambers)
    if (chamber.station() == station && chamber.detector() == detector)
      numSegments += chamber.segments.size();
  return numSegments;
}

int TrackDetMatchInfo::numberOfSegmentsInDetector(int detector) const {
  int numSegments = 0;
  for (const auto& chamber : chambers)
    if (chamber.detector() == detector)
      numSegments += chamber.segments.size();
  return numSegments;
}
