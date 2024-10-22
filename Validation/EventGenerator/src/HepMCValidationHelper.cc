#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <limits>
#include "TLorentzVector.h"

//#define DEBUG_HepMCValidationHelper

namespace HepMCValidationHelper {
  void findFSRPhotons(const std::vector<const HepMC::GenParticle*>& leptons,
                      const HepMC::GenEvent* all,
                      double deltaRcut,
                      std::vector<const HepMC::GenParticle*>& fsrphotons) {
    std::vector<const HepMC::GenParticle*> status1;
    allStatus1(all, status1);
    findFSRPhotons(leptons, status1, deltaRcut, fsrphotons);
  }

  void findFSRPhotons(const std::vector<const HepMC::GenParticle*>& leptons,
                      const std::vector<const HepMC::GenParticle*>& all,
                      double deltaRcut,
                      std::vector<const HepMC::GenParticle*>& fsrphotons) {
    //find all status 1 photons
    std::vector<const HepMC::GenParticle*> allphotons;
    for (unsigned int i = 0; i < all.size(); ++i) {
      if (all[i]->status() == 1 && all[i]->pdg_id() == 22)
        allphotons.push_back(all[i]);
    }

    //loop over the photons and check the distance wrt the leptons
    for (unsigned int ipho = 0; ipho < allphotons.size(); ++ipho) {
      bool close = false;
      for (unsigned int ilep = 0; ilep < leptons.size(); ++ilep) {
        if (deltaR(allphotons[ipho]->momentum(), leptons[ilep]->momentum()) < deltaRcut) {
          close = true;
          break;
        }
      }
      if (close)
        fsrphotons.push_back(allphotons[ipho]);
    }
  }

  //returns true if a status 3 particle is a tau or if a status 1 particle is either an electron or a neutrino
  bool isChargedLepton(const HepMC::GenParticle* part) {
    int status = part->status();
    unsigned int pdg_id = abs(part->pdg_id());
    if (status == 2)
      return pdg_id == 15;
    else
      return status == 1 && (pdg_id == 11 || pdg_id == 13);
  }

  //returns true if a status 1 particle is a neutrino
  bool isNeutrino(const HepMC::GenParticle* part) {
    int status = part->status();
    unsigned int pdg_id = abs(part->pdg_id());
    return status == 1 && (pdg_id == 12 || pdg_id == 14 || pdg_id == 16);
  }

  //returns true is status 3 particle is tau
  bool isTau(const HepMC::GenParticle* part) { return part->status() == 2 && abs(part->pdg_id()) == 15; }
  /* 
  void getTaus(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& taus){
    for (HepMC::GenEvent::particle_const_iterator iter = all->particles_begin(); iter != all->particles_end(); ++iter){
      if (abs((*iter)->pdg_id()) == 15) taus.push_back(*iter);
    }
  }
*/
  // get all status 1 particles
  void allStatus1(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status1) {
    for (HepMC::GenEvent::particle_const_iterator iter = all->particles_begin(); iter != all->particles_end(); ++iter) {
      if ((*iter)->status() == 1)
        status1.push_back(*iter);
    }
  }

  void allStatus2(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status1) {
    for (HepMC::GenEvent::particle_const_iterator iter = all->particles_begin(); iter != all->particles_end(); ++iter) {
      if ((*iter)->status() == 2)
        status1.push_back(*iter);
    }
  }

  void allStatus3(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& status1) {
    for (HepMC::GenEvent::particle_const_iterator iter = all->particles_begin(); iter != all->particles_end(); ++iter) {
      if ((*iter)->status() == 3)
        status1.push_back(*iter);
    }
  }

  void findDescendents(const HepMC::GenParticle* a, std::vector<const HepMC::GenParticle*>& descendents) {
    HepMC::GenVertex* decayVertex = a->end_vertex();
    if (!decayVertex)
      return;
    HepMC::GenVertex::particles_out_const_iterator ipart;
    for (ipart = decayVertex->particles_out_const_begin(); ipart != decayVertex->particles_out_const_end(); ++ipart) {
      if ((*ipart)->status() == 1)
        descendents.push_back(*ipart);
      else
        findDescendents(*ipart, descendents);
    }
  }

  void removeIsolatedLeptons(const HepMC::GenEvent* all,
                             double deltaRcut,
                             double sumPtCut,
                             std::vector<const HepMC::GenParticle*>& pruned) {
    //get all status 1 particles
    std::vector<const HepMC::GenParticle*> status1;
    allStatus1(all, status1);
    std::vector<const HepMC::GenParticle*> toRemove;
    //loop on all particles and find candidates to be isolated
    for (unsigned int i = 0; i < status1.size(); ++i) {
      //if it is a neutrino is a charged lepton (not a tau) this is a candidate to be isolated
      if (isNeutrino(status1[i]) || (isChargedLepton(status1[i]) && !isTau(status1[i]))) {
        //list of particles not to be considered in the isolation computation.
        //this includes the particle to be isolated and the fsr photons in case of charged lepton
        std::vector<const HepMC::GenParticle*> leptons;
        leptons.push_back(status1[i]);
        std::vector<const HepMC::GenParticle*> removedForIsolation;
        removedForIsolation.push_back(status1[i]);
        if (isChargedLepton(status1[i]))
          findFSRPhotons(leptons, status1, deltaRcut, removedForIsolation);
#ifdef DEBUG_HepMCValidationHelper
          //std::cout << removedForIsolation.size() << " particles to be removed for isolation calculation " << std::endl;
#endif
        //create vector of particles to compute isolation (removing removedForIsolation);
        std::vector<const HepMC::GenParticle*> forIsolation;
        std::vector<const HepMC::GenParticle*>::iterator iiso;
        for (iiso = status1.begin(); iiso != status1.end(); ++iiso) {
          std::vector<const HepMC::GenParticle*>::const_iterator iremove;
          bool marked = false;
          for (iremove = removedForIsolation.begin(); iremove != removedForIsolation.end(); ++iremove) {
            if ((*iiso)->barcode() == (*iremove)->barcode()) {
#ifdef DEBUG_HepMCValidationHelper
              //std::cout << "removing particle " << **iiso << " from the list of particles to compute isolation" << std::endl;
#endif
              marked = true;
              break;
            }
          }
          if (!marked)
            forIsolation.push_back(*iiso);
        }
        //now compute isolation
        double sumIso = 0;
        for (iiso = forIsolation.begin(); iiso < forIsolation.end(); ++iiso) {
          if (deltaR(leptons.front()->momentum(), (*iiso)->momentum()) < deltaRcut) {
            sumIso += (*iiso)->momentum().perp();
          }
        }
        //if isolated remove from the pruned list
        if (sumIso < sumPtCut) {
#ifdef DEBUG_HepMCValidationHelper
          std::cout << "particle " << *status1[i] << " is considered isolated, with sumPt " << sumIso << std::endl;
#endif
          toRemove.insert(toRemove.end(), removedForIsolation.begin(), removedForIsolation.end());
        }
#ifdef DEBUG_HepMCValidationHelper
        else {
          std::cout << "NOT isolated! " << *status1[i] << " is considered not isolated, with sumPt " << sumIso
                    << std::endl;
        }
#endif
      }
    }
    //at this point we have taken care of the electrons and muons, but pruned could  still contain the decay products of isolated taus,
    //we want to remove these as well
    std::vector<const HepMC::GenParticle*> status2;
    allStatus2(all, status2);
    std::vector<const HepMC::GenParticle*> taus;
    //getTaus(all, taus);
    for (unsigned int i = 0; i < status2.size(); ++i) {
      if (isTau(status2[i])) {
        //check the list we have already for duplicates
        //there use to be duplicates in some generators (sherpa)
        bool duplicate = false;
        TLorentzVector taumomentum(status2[i]->momentum().x(),
                                   status2[i]->momentum().y(),
                                   status2[i]->momentum().z(),
                                   status2[i]->momentum().t());
        for (unsigned int j = 0; j < taus.size(); ++j) {
          //compare momenta
          TLorentzVector othermomentum(
              taus[j]->momentum().x(), taus[j]->momentum().y(), taus[j]->momentum().z(), taus[j]->momentum().t());
          othermomentum -= taumomentum;
          if (status2[i]->pdg_id() == taus[j]->pdg_id() &&
              othermomentum.E() < 0.1 &&  //std::numeric_limits<float>::epsilon() &&
              othermomentum.P() < 0.1) {  //std::numeric_limits<float>::epsilon()){
            duplicate = true;
            break;
          }
        }
        if (!duplicate)
          taus.push_back(status2[i]);
      }
    }
    //loop over the taus, find the descendents, remove all these from the list of particles to compute isolation
    for (unsigned int i = 0; i < taus.size(); ++i) {
      std::vector<const HepMC::GenParticle*> taudaughters;
      findDescendents(taus[i], taudaughters);
      if (taudaughters.empty()) {
        std::ostringstream ss;
        auto vertex = taus[i]->end_vertex();
        if (vertex) {
          ss << "( " << vertex->point3d().x() << " " << vertex->point3d().y() << " " << vertex->point3d().z() << " )";
        } else {
          ss << "( did not decay )";
        }
        throw cms::Exception("NoTauDaugters")
            << " HepMCValidationHelper found no daughters for Tau within index " << i << " and info \n"
            << *taus[i] << " decay point " << ss.str()
            << "\n  This should not be able to happen and needs to be fixed.";
      }
      const HepMC::FourVector& taumom = taus[i]->momentum();
      //remove the daughters from the list of particles to compute isolation
      std::vector<const HepMC::GenParticle*> forIsolation;
      std::vector<const HepMC::GenParticle*>::iterator iiso;
      for (iiso = status1.begin(); iiso < status1.end(); ++iiso) {
        bool marked = false;
        std::vector<const HepMC::GenParticle*>::const_iterator iremove;
        for (iremove = taudaughters.begin(); iremove != taudaughters.end(); ++iremove) {
          if ((*iiso)->barcode() == (*iremove)->barcode()) {
#ifdef DEBUG_HepMCValidationHelper
//            std::cout << "removing particle " << **iiso << " from the list of particles to compute isolation because it comes from a tau" << std::endl;
#endif
            marked = true;
            break;
          }
        }
        if (!marked)
          forIsolation.push_back(*iiso);
      }
      //no compute isolation wrt the status 2 tau direction
      double sumIso = 0;
      for (iiso = forIsolation.begin(); iiso < forIsolation.end(); ++iiso) {
        if (deltaR(taumom, (*iiso)->momentum()) < deltaRcut) {
          sumIso += (*iiso)->momentum().perp();
        }
      }
      //if isolated remove the tau daughters from the pruned list
      if (sumIso < sumPtCut) {
#ifdef DEBUG_HepMCValidationHelper
        std::cout << "particle " << *taus[i] << " is considered isolated, with sumPt " << sumIso << std::endl;
#endif
        toRemove.insert(toRemove.end(), taudaughters.begin(), taudaughters.end());
      }
    }

    //now actually remove
    pruned.clear();
    for (unsigned int i = 0; i < status1.size(); ++i) {
      bool marked = false;
      std::vector<const HepMC::GenParticle*>::const_iterator iremove;
      for (iremove = toRemove.begin(); iremove != toRemove.end(); ++iremove) {
        if (status1[i]->barcode() == (*iremove)->barcode()) {
          marked = true;
          break;
        }
      }
      if (!marked)
        pruned.push_back(status1[i]);
    }

#ifdef DEBUG_HepMCValidationHelper
    std::cout << "list of remaining particles:" << std::endl;
    for (unsigned int i = 0; i < pruned.size(); ++i) {
      std::cout << *pruned[i] << std::endl;
    }
#endif
  }

  //get all visible status1 particles
  void allVisibleParticles(const HepMC::GenEvent* all, std::vector<const HepMC::GenParticle*>& visible) {
    std::vector<const HepMC::GenParticle*> status1;
    visible.clear();
    allStatus1(all, status1);
    for (unsigned int i = 0; i < status1.size(); ++i) {
      if (!isNeutrino(status1[i]))
        visible.push_back(status1[i]);
    }
  }

  //compute generated met
  TLorentzVector genMet(const HepMC::GenEvent* all, double etamin, double etamax) {
    std::vector<const HepMC::GenParticle*> visible;
    allVisibleParticles(all, visible);
    TLorentzVector momsum(0., 0., 0., 0.);
    for (unsigned int i = 0; i < visible.size(); ++i) {
      if (visible[i]->momentum().eta() > etamin && visible[i]->momentum().eta() < etamax) {
        TLorentzVector mom(visible[i]->momentum().x(),
                           visible[i]->momentum().y(),
                           visible[i]->momentum().z(),
                           visible[i]->momentum().t());
        momsum += mom;
      }
    }
    TLorentzVector met(-momsum.Px(), -momsum.Py(), 0., momsum.Pt());
    return met;
  }
}  // namespace HepMCValidationHelper
