#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace SimGeneral_TrackingAnalysis {
  struct dictionary {
    edm::Ptr<TrackingParticle> TP;
    std::vector<edm::Ptr<TrackingParticle>> V_TP;
    std::pair<TrackingParticleRef, TrackPSimHitRef> dummy13;
    edm::Wrapper<std::pair<TrackingParticleRef, TrackPSimHitRef>> dummy14;
    std::vector<std::pair<TrackingParticleRef, TrackPSimHitRef>> dummy07;
    edm::Wrapper<std::vector<std::pair<TrackingParticleRef, TrackPSimHitRef>>> dummy08;

    std::pair<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, edm::Ptr<TrackingParticle>> P_TAM_S_TP_PD;
    std::pair<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>>> P_TAM_TP_S_PD;

    std::map<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, edm::Ptr<TrackingParticle>> M_TAM_S_TP_PD;
    std::map<edm::Ptr<TrackingParticle>, std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>>> M_TAM_TP_S_PD;
  };
}  // namespace SimGeneral_TrackingAnalysis
