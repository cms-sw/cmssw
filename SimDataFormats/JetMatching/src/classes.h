#include <utility>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

namespace {
  namespace {
    reco::JetFlavour                                 jf;
    reco::JetFlavourMatchingCollectionBase           jfmcb;
    reco::JetFlavourMatchingCollection               jfmc;
    reco::JetFlavourMatchingRef                      jfmr;
    reco::JetFlavourMatchingRefProd                  jfmrp;
    reco::JetFlavourMatchingRefVector                jfrv;
    edm::Wrapper<reco::JetFlavourMatchingCollection> wjfmc;
  }
}
