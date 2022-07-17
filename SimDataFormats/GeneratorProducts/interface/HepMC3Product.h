#ifndef SimDataFormats_GeneratorProducts_HepMC3Product_h
#define SimDataFormats_GeneratorProducts_HepMC3Product_h

/** \class HepMC3Product
 *
 *  \author Joanna Weng, Filip Moortgat, Mikhail Kirsanov
 */

#include "DataFormats/Common/interface/Ref.h"
#include <TMatrixD.h>
#include <HepMC3/GenEvent.h>
#include <cstddef>

namespace HepMC3 {
  class FourVector;
  class GenParticle;
  class GenVertex;
}  // namespace HepMC3

namespace edm {
  class HepMC3Product {
  public:
    HepMC3Product() : evt_(nullptr), isVtxGenApplied_(false), isVtxBoostApplied_(false), isPBoostApplied_(false) {}

    explicit HepMC3Product(HepMC3::GenEvent *evt);
    virtual ~HepMC3Product();

    void addHepMCData(HepMC3::GenEvent *evt);

    void applyVtxGen(HepMC3::FourVector const *vtxShift) { applyVtxGen(*vtxShift); }
    void applyVtxGen(HepMC3::FourVector const &vtxShift);

    void boostToLab(TMatrixD const *lorentz, std::string const &type);

    const HepMC3::GenEvent &getHepMCData() const;

    const HepMC3::GenEvent *GetEvent() const { return evt_; }

    bool isVtxGenApplied() const { return isVtxGenApplied_; }
    bool isVtxBoostApplied() const { return isVtxBoostApplied_; }
    bool isPBoostApplied() const { return isPBoostApplied_; }

    HepMC3Product(HepMC3Product const &orig);
    HepMC3Product &operator=(HepMC3Product const &other);
    HepMC3Product(HepMC3Product &&orig);
    HepMC3Product &operator=(HepMC3Product &&other);
    void swap(HepMC3Product &other);

  private:
    HepMC3::GenEvent *evt_;

    bool isVtxGenApplied_;
    bool isVtxBoostApplied_;
    bool isPBoostApplied_;
  };

  // This allows edm::Refs to work with HepMC3Product
  namespace refhelper {
    template <>
    struct FindTrait<edm::HepMC3Product, HepMC3::GenParticle> {
      struct Find {
        using first_argument_type = edm::HepMC3Product const &;
        using second_argument_type = int;
        using result_type = HepMC3::GenParticle const *;

        result_type operator()(first_argument_type iContainer, second_argument_type iBarCode) {
          //return iContainer.getHepMCData().barcode_to_particle(iBarCode);
          return nullptr;
        }
      };

      typedef Find value;
    };

    template <>
    struct FindTrait<edm::HepMC3Product, HepMC3::GenVertex> {
      struct Find {
        using first_argument_type = edm::HepMC3Product const &;
        using second_argument_type = int;
        using result_type = HepMC3::GenVertex const *;

        result_type operator()(first_argument_type iContainer, second_argument_type iBarCode) {
          //return iContainer.getHepMCData().barcode_to_vertex(iBarCode);
          return nullptr;
        }
      };

      typedef Find value;
    };
  }  // namespace refhelper
}  // namespace edm

#endif  // SimDataFormats_GeneratorProducts_HepMC3Product_h
