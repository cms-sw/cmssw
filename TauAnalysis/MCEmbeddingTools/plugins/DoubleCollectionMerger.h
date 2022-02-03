/** \class DoubleCollectionMerger
 *
 * 
 * \author Per Ahrens
 *
 * 
 *
 * 
 *
 */
#ifndef TauAnalysis_MCEmbeddingTools_DoubleCollectionMerger_H
#define TauAnalysis_MCEmbeddingTools_DoubleCollectionMerger_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


//#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"
#include <string>
#include <iostream>
#include <map>



template <typename T1, typename T2, typename T3, typename T4>
class DoubleCollectionMerger : public  edm::stream::EDProducer<>
{
 public:
  explicit DoubleCollectionMerger(const edm::ParameterSet&);
  ~DoubleCollectionMerger();

 private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef T1 MergeCollection1;
  typedef T2 BaseHit1;
  typedef T3 MergeCollection2;
  typedef T4 BaseHit2;
  std::map<std::string,  std::vector<edm::EDGetTokenT<MergeCollection1 > > > inputs1_;
  std::map<std::string,  std::vector<edm::EDGetTokenT<MergeCollection2 > > > inputs2_;

  void fill_output_obj(std::unique_ptr<MergeCollection1 > & output1, std::vector<edm::Handle<MergeCollection1> > &inputCollections1);
  void fill_output_obj(std::unique_ptr<MergeCollection2 > & output2, std::vector<edm::Handle<MergeCollection2> > &inputCollections2);
  void fill_output_obj_digiflag(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections);
  void fill_output_obj_digiflag(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections);

};

template <typename T1, typename T2, typename T3, typename T4>
DoubleCollectionMerger<T1,T2,T3,T4>::DoubleCollectionMerger(const edm::ParameterSet& iConfig)
{
  std::vector<edm::InputTag> inCollections =  iConfig.getParameter<std::vector<edm::InputTag> >("mergCollections");
  for (auto const & inCollection : inCollections){
    inputs1_[inCollection.instance()].push_back(consumes<MergeCollection1 >(inCollection));
    inputs2_[inCollection.instance()].push_back(consumes<MergeCollection2 >(inCollection));
  }
  for (auto toproduce : inputs1_){
    //std::cout<<toproduce.first<<"\t"<<toproduce.second.size()<<std::endl;
    produces<MergeCollection1>(toproduce.first);
  }
  for (auto toproduce : inputs2_){
    //std::cout<<toproduce.first<<"\t"<<toproduce.second.size()<<std::endl;
    produces<MergeCollection2>(toproduce.first);
  }
}


template <typename T1, typename T2, typename T3, typename T4>
DoubleCollectionMerger<T1,T2,T3,T4>::~DoubleCollectionMerger()
{
// nothing to be done yet...  
}


template <typename T1, typename T2, typename T3, typename T4>
void DoubleCollectionMerger<T1,T2,T3,T4>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //std::cout << "DoubleCollectionMerger<T1,T2,T3,T4>::produce" << std::endl;
  for (auto input_ : inputs1_){
    //std::cout << "input_.first()=" << input_.first << std::endl;
    //std::cout << "input_.second.size()=" << input_.second.size() << std::endl;
    std::unique_ptr<MergeCollection1 > output(new MergeCollection1());
    std::vector<edm::Handle<MergeCollection1> > inputCollections;
    inputCollections.resize(input_.second.size());
    for (unsigned id = 0; id<input_.second.size(); id++){
      //std::cout << "input_.second[id]=" << input_.second[id] << std::endl;
      //std::cout << "input_.second[id]=" << id << std::endl;
      iEvent.getByToken(input_.second[id], inputCollections[id]);
     }
    fill_output_obj(output,inputCollections);
    iEvent.put(std::move(output),input_.first);
  
  }

  for (auto input_ : inputs2_){
    //std::cout << "input_.first()=" << input_.first << std::endl;
    //std::cout << "input_.second.size()=" << input_.second.size() << std::endl;
    std::unique_ptr<MergeCollection2 > output(new MergeCollection2());
    std::vector<edm::Handle<MergeCollection2> > inputCollections;
    inputCollections.resize(input_.second.size());
    for (unsigned id = 0; id<input_.second.size(); id++){
      //std::cout << "input_.second[id]=" << input_.second[id] << std::endl;
      //std::cout << "input_.second[id]=" << id << std::endl;
      iEvent.getByToken(input_.second[id], inputCollections[id]);
     }
    fill_output_obj(output,inputCollections);
    iEvent.put(std::move(output),input_.first);  
  }
}
#endif
