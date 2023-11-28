/** \class DoubleCollectionMerger
 *
 * \author Per Ahrens
 *
 */
#ifndef TauAnalysis_MCEmbeddingTools_DoubleCollectionMerger_H
#define TauAnalysis_MCEmbeddingTools_DoubleCollectionMerger_H

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/SortedCollection.h"

// #include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"
#include <iostream>
#include <map>
#include <string>

template <typename T1, typename T2, typename T3, typename T4>
class DoubleCollectionMerger : public edm::stream::EDProducer<> {
public:
  explicit DoubleCollectionMerger(const edm::ParameterSet &);
  ~DoubleCollectionMerger();

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef T1 MergeCollection1;
  typedef T2 BaseHit1;
  typedef T3 MergeCollection2;
  typedef T4 BaseHit2;
  std::map<std::string, std::vector<edm::EDGetTokenT<MergeCollection1>>> inputs1_;
  std::map<std::string, std::vector<edm::EDGetTokenT<MergeCollection2>>> inputs2_;

  void fill_output_obj(std::unique_ptr<MergeCollection1> &output1,
                       std::vector<edm::Handle<MergeCollection1>> &inputCollections1);
  void fill_output_obj(std::unique_ptr<MergeCollection2> &output2,
                       std::vector<edm::Handle<MergeCollection2>> &inputCollections2);
  void fill_output_obj_digiflag(std::unique_ptr<MergeCollection1> &output,
                                std::vector<edm::Handle<MergeCollection1>> &inputCollections);
  void fill_output_obj_digiflag(std::unique_ptr<MergeCollection2> &output,
                                std::vector<edm::Handle<MergeCollection2>> &inputCollections);
  void fill_output_obj_hcaldigi(std::unique_ptr<MergeCollection1> &output,
                                std::vector<edm::Handle<MergeCollection1>> &inputCollections);
  void fill_output_obj_hcaldigi(std::unique_ptr<MergeCollection2> &output,
                                std::vector<edm::Handle<MergeCollection2>> &inputCollections);
};
#endif
