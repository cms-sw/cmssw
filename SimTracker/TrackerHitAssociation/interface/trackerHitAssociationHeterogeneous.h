#ifndef SimTracker_TrackerHitAssociation_plugins_trackerHitAssociationHeterogeneousProduct_h
#define SimTracker_TrackerHitAssociation_plugins_trackerHitAssociationHeterogeneousProduct_h

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace trackerHitAssociationHeterogeneous {

  struct ClusterSLView {
    using Clus2TP = std::array<uint32_t, 7>;

    Clus2TP* links_d;
    uint32_t* tkId_d;
    uint32_t* tkId2_d;
    uint32_t* n1_d;
    uint32_t* n2_d;
  };

  template <typename Traits>
  class Product {
  public:
    template <typename T>
    using unique_ptr = typename Traits::template unique_ptr<T>;

    Product() = default;
    ~Product() = default;
    Product(Product const&) = delete;
    Product(Product&&) = default;

    Product(int nlinks, int nhits, cudaStream_t stream);

    ClusterSLView& view() { return m_view; }
    ClusterSLView const& view() const { return m_view; }

    int nLinks() const { return m_nLinks; }
    int nHits() const { return m_nHits; }

  private:
    static constexpr uint32_t n32 = 4;

    unique_ptr<uint32_t[]> m_storeTP;  //!
    unique_ptr<uint32_t[]> m_store32;  //!

    ClusterSLView m_view;  //!

    int m_nLinks;
    int m_nHits;
  };

  template <typename Traits>
  Product<Traits>::Product(int nlinks, int nhits, cudaStream_t stream) : m_nLinks(nlinks), m_nHits(nhits) {
    m_storeTP = Traits::template make_device_unique<uint32_t[]>(m_nLinks * 7, stream);
    m_store32 = Traits::template make_device_unique<uint32_t[]>(m_nHits * n32, stream);

    auto get32 = [&](int i) { return m_store32.get() + i * m_nHits; };

    m_view.links_d = (ClusterSLView::Clus2TP*)(m_storeTP.get());
    m_view.tkId_d = get32(0);
    m_view.tkId2_d = get32(1);
    m_view.n1_d = get32(2);
    m_view.n2_d = get32(3);
  }

  using ProductCUDA = Product<cms::cudacompat::GPUTraits>;

}  // namespace trackerHitAssociationHeterogeneous

#endif  // SimTracker_TrackerHitAssociation_plugins_trackerHitAssociationHeterogeneousProduct_h
