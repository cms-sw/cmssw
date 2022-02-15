#include "Validation/EventGenerator/interface/GenPtcValidationHelper.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <limits>
#include "DataFormats/Math/interface/LorentzVector.h"

namespace GenPtcValidationHelper {
  bool isFinalStateLepton(const reco::GenParticleRef ptc) {
    bool isLepton = ( std::abs(ptc->pdgId())==11 || std::abs(ptc->pdgId())==13 );
    bool isFinalState = ( ptc->status() == 1 );
    return isLepton && isFinalState;
  }

  void findFSRPhotons(const std::vector<reco::GenParticleRef>& leps, const reco::GenParticleCollection& ptcls,
    double dRcut, std::vector<reco::GenParticle>& FSRphotons) {
      for (unsigned iptc = 0; iptc < ptcls.size(); iptc++) {
        reco::GenParticle ptc = ptcls.at(iptc);
        if (ptc.status()==1 && ptc.pdgId()==22) {
          math::XYZTLorentzVector phoMom(ptc.px(),ptc.py(),ptc.pz(),ptc.p());
          bool close = false;
          for (unsigned ilep = 0; ilep < leps.size(); ilep++) {
            reco::GenParticleRef lep = leps.at(ilep);
            math::XYZTLorentzVector lepMom(lep->px(),lep->py(),lep->pz(),lep->p());
            if ( reco::deltaR2(lepMom,phoMom) < dRcut*dRcut ) {
              close = true;
              break;
            }
          }
          if (close) FSRphotons.push_back(ptc);
        }
      }
  }

}
