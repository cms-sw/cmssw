#include "TauAnalysis/MCEmbeddingTools/plugins/DoubleCollectionMerger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/SortedCollection.h"

#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"
#include "DataFormats/EcalDigi/interface/EESrFlag.h"
#include "DataFormats/EcalDigi/interface/EBSrFlag.h"
#include "DataFormats/EcalDigi/src/EcalSrFlag.cc"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"


typedef DoubleCollectionMerger<edm::SortedCollection<EESrFlag>, EESrFlag, edm::SortedCollection<EBSrFlag>, EBSrFlag> EcalSrFlagColMerger;
typedef DoubleCollectionMerger<HBHEDigiCollection, HBHEDataFrame, HcalCalibDigiCollection, HcalCalibDataFrame> HcalDigiColMerger;

// Here some overloaded functions, which are needed such that the right merger function is called for the indivudal Collections
template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj_digiflag(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections)
{
  std::map<uint32_t, BaseHit1>   output_map;

  // First merge the two collections again
  for (auto const & inputCollection : inputCollections){
    for ( typename MergeCollection1::const_iterator obj = inputCollection->begin(); obj!= inputCollection->end(); ++obj ) {
      DetId detIdObject( obj->id().rawId() );

      std::map<uint32_t, EESrFlag>::iterator it = output_map.find(detIdObject.rawId());
      if (it == output_map.end()) {
        BaseHit1 *akt_flag_obj = &output_map[detIdObject.rawId()];
        T2 newSrFlag(*obj);
        *akt_flag_obj = newSrFlag;
      } else {
        //re-determine flag
        BaseHit1 preFlag = it->second;
        BaseHit1 *akt_flag_obj = &output_map[detIdObject.rawId()];
        T2 newSrFlag(*obj);

        newSrFlag.setValue(std::max(obj->value(),preFlag.value()));
        if (preFlag.value() == 3 or obj->value() == 3) newSrFlag.setValue(3);
        if (preFlag.value() == 7 or obj->value() == 7) newSrFlag.setValue(7);

        *akt_flag_obj = newSrFlag;
      }
    }
  }

  // Now save it into the standard CMSSW format
  for (typename std::map<uint32_t, BaseHit1 >::const_iterator outFlags = output_map.begin(); outFlags != output_map.end(); ++outFlags ) {
    BaseHit1 currFlag = outFlags->second;
    output->push_back(outFlags->second);
  }
  output->sort(); //Do a sort for this collection
}

template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj_digiflag(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections)
{
  std::map<uint32_t, BaseHit2>   output_map;

  // First merge the two collections again
  for (auto const & inputCollection : inputCollections){
    for ( typename MergeCollection2::const_iterator obj = inputCollection->begin(); obj!= inputCollection->end(); ++obj ) {
      DetId detIdObject( obj->id().rawId() );

      std::map<uint32_t, EBSrFlag>::iterator it = output_map.find(detIdObject.rawId());
      if (it == output_map.end()) {
        BaseHit2 *akt_flag_obj = &output_map[detIdObject.rawId()];
        T4 newSrFlag(*obj);
        *akt_flag_obj = newSrFlag;
      } else {
        //re-determine flag
        BaseHit2 preFlag = it->second;
        BaseHit2 *akt_flag_obj = &output_map[detIdObject.rawId()];
        T4 newSrFlag(*obj);

        newSrFlag.setValue(std::max(obj->value(),preFlag.value()));
        if (preFlag.value() == 3 or obj->value() == 3) newSrFlag.setValue(3);
        if (preFlag.value() == 7 or obj->value() == 7) newSrFlag.setValue(7);

        *akt_flag_obj = newSrFlag;
      }
    }
  }

  // Now save it into the standard CMSSW format
  for (typename std::map<uint32_t, BaseHit2 >::const_iterator outFlags = output_map.begin(); outFlags != output_map.end(); ++outFlags ) {
    BaseHit2 currFlag = outFlags->second;
    output->push_back(outFlags->second);
  }
  output->sort(); //Do a sort for this collection
}

template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj_hcaldigi(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections)
{
  //TODO: implement proper merging, only skeleton for the time-being
  return;
}

template <typename T1, typename T2, typename T3, typename T4>
void  DoubleCollectionMerger<T1,T2,T3,T4>::fill_output_obj_hcaldigi(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections)
{
  //TODO: implement proper merging, only skeleton for the time-being
  return;
}

template <>
void  DoubleCollectionMerger<edm::SortedCollection<EESrFlag>, EESrFlag, edm::SortedCollection<EBSrFlag>, EBSrFlag>::fill_output_obj(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections)
{
  fill_output_obj_digiflag(output,inputCollections);
}

template <>
void  DoubleCollectionMerger<edm::SortedCollection<EESrFlag>, EESrFlag, edm::SortedCollection<EBSrFlag>, EBSrFlag>::fill_output_obj(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections)
{
  fill_output_obj_digiflag(output,inputCollections);
}

template <>
void  DoubleCollectionMerger<HBHEDigiCollection, HBHEDataFrame, HcalCalibDigiCollection, HcalCalibDataFrame>::fill_output_obj(std::unique_ptr<MergeCollection1 > & output, std::vector<edm::Handle<MergeCollection1> > &inputCollections)
{
  fill_output_obj_hcaldigi(output,inputCollections);
}

template <>
void  DoubleCollectionMerger<HBHEDigiCollection, HBHEDataFrame, HcalCalibDigiCollection, HcalCalibDataFrame>::fill_output_obj(std::unique_ptr<MergeCollection2 > & output, std::vector<edm::Handle<MergeCollection2> > &inputCollections)
{
  fill_output_obj_hcaldigi(output,inputCollections);
}


DEFINE_FWK_MODULE(EcalSrFlagColMerger);
DEFINE_FWK_MODULE(HcalDigiColMerger);
