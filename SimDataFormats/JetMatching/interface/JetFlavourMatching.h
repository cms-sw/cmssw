#ifndef SimDataFormats_JetMatching_JetFlavourMatching
#define SimDataFormats_JetMatching_JetFlavourMatching
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>

typedef  edm::AssociationVector<reco::CaloJetRefProd,std::vector<int> >  JetFlavourMatchingCollection;

typedef  JetFlavourMatchingCollection::value_type JetFlavourMatching;

typedef  edm::Ref<JetFlavourMatchingCollection> JetFlavourMatchingRef;
   
typedef  edm::RefProd<JetFlavourMatchingCollection> JetFlavourMatchingRefProd;
   
typedef  edm::RefVector<JetFlavourMatchingCollection> JetFlavourMatchingRefVector; 

#endif
