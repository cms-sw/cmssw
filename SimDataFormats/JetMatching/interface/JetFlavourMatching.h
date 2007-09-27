#ifndef SimDataFormats_JetMatching_JetFlavourMatching
#define SimDataFormats_JetMatching_JetFlavourMatching
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <vector>

typedef edm::AssociationVector<reco::CaloJetRefProd,std::vector<int> > JetFlavourMatchingCollectionBase;

class JetFlavourMatchingCollection : public JetFlavourMatchingCollectionBase {
public:
  JetFlavourMatchingCollection() :
    JetFlavourMatchingCollectionBase()
  { }
 
  JetFlavourMatchingCollection(const reco::CaloJetRefProd & ref) :
    JetFlavourMatchingCollectionBase(ref)
  { }

  JetFlavourMatchingCollection(const JetFlavourMatchingCollectionBase &v) :
    JetFlavourMatchingCollectionBase(v)
  { }
};

typedef  JetFlavourMatchingCollection::value_type JetFlavourMatching;

typedef  edm::Ref<JetFlavourMatchingCollection> JetFlavourMatchingRef;
   
typedef  edm::RefProd<JetFlavourMatchingCollection> JetFlavourMatchingRefProd;
   
typedef  edm::RefVector<JetFlavourMatchingCollection> JetFlavourMatchingRefVector; 

#endif
