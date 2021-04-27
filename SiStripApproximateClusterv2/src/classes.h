#ifndef SISTRIPAPPROXIMATECLUSTERv2_CLASSES_H
#define SISTRIPAPPROXIMATECLUSTERv2_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripApproximateClusterv2/interface/SiStripApproximateClusterv2.h"
#include "DataFormats/Common/interface/ContainerMask.h"

namespace DataFormats_SiStripApproximateClusterv2 {
  struct dictionary2 {


    edmNew::DetSetVector<SiStripApproximateClusterv2> dsvn;

    edm::Wrapper< SiStripApproximateClusterv2 > dummy0;
    edm::Wrapper< std::vector<SiStripApproximateClusterv2>  > dummy1;

    edm::Wrapper< edmNew::DetSetVector<SiStripApproximateClusterv2> > dummy4_bis;
    
    edm::Wrapper<edm::ContainerMask<edmNew::DetSetVector<SiStripApproximateClusterv2> > > dummy_w_cm1;

    std::vector<edm::Ref<edmNew::DetSetVector<SiStripApproximateClusterv2>,SiStripApproximateClusterv2,edmNew::DetSetVector<SiStripApproximateClusterv2>::FindForDetSetVector> > dummy_v;
    edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<SiStripApproximateClusterv2>,SiStripApproximateClusterv2,edmNew::DetSetVector<SiStripApproximateClusterv2>::FindForDetSetVector> > dumm_dtvr;
    edm::Wrapper<edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<SiStripApproximateClusterv2>,SiStripApproximateClusterv2,edmNew::DetSetVector<SiStripApproximateClusterv2>::FindForDetSetVector> > > dumm_dtvr_w;


    edm::Ref<edmNew::DetSetVector<SiStripApproximateClusterv2>, SiStripApproximateClusterv2, edmNew::DetSetVector<SiStripApproximateClusterv2>::FindForDetSetVector > refNew;
  };
}


#endif // SISTRIPAPPROXIMATECLUSTER_CLASSES_H
