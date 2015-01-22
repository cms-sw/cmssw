#ifndef SimDataFormats_Associations_MuonTrackType_h
#define SimDataFormats_Associations_MuonTrackType_h
// -*- C++ -*-
//
// Package:     SimDataFormats/Associations
// Class  :     MuonTrackType
// 
/**\enum MuonTrackType MuonTrackType.h "SimDataFormats/Associations/interface/MuonTrackType.h"

 Description: Types of muon tracks used by MuonToSimAssociator

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 07 Jan 2015 21:03:32 GMT
//

#include <vector>
#include "DataFormats/Common/interface/RefToBase.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/MuonReco/interface/Muon.h"

namespace reco {
    enum MuonTrackType { InnerTk, OuterTk, GlobalTk, Segments };

    struct RefToBaseSort { 
      template<typename T> bool operator()(const edm::RefToBase<T> &r1, const edm::RefToBase<T> &r2) const { 
        return (r1.id() == r2.id() ? r1.key() < r2.key() : r1.id() < r2.id()); 
      }
    };
    typedef std::map<edm::RefToBase<reco::Muon>, std::vector<std::pair<TrackingParticleRef, double> >, RefToBaseSort> MuonToSimCollection;
    typedef std::map<TrackingParticleRef, std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > >               SimToMuonCollection;

}

#endif
