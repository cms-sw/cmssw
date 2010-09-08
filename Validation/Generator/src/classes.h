#include "DataFormats/JetReco/interface/CaloJetCollection.h" 

#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"


namespace {
  struct dictionary {
    edm::RefToBaseVector<reco::CaloJet> jrtbv;
    edm::Wrapper<edm::RefToBaseVector<reco::CaloJet> > jrtbv_w;
    edm::reftobase::BaseVectorHolder<reco::CaloJet> * bvhj_p;
  };
}

