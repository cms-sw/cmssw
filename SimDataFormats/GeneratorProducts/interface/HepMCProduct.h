#ifndef SimDataFormats_GeneratorProducts_HepMCProduct_h
#define SimDataFormats_GeneratorProducts_HepMCProduct_h

/** \class HepMCProduct
 *
 *  \author Joanna Weng, Filip Moortgat
 */

#include "DataFormats/Common/interface/Ref.h"
#include <TMatrixD.h>
#include <HepMC/GenEvent.h>
#include <cstddef>

namespace HepMC {
  class FourVector;
  class GenParticle;
  class GenVertex;
}  // namespace HepMC

namespace edm {
  class HepMCProduct {
  public:
    HepMCProduct() : evt_(nullptr), isVtxGenApplied_(false), isVtxBoostApplied_(false), isPBoostApplied_(false) {}

    explicit HepMCProduct(HepMC::GenEvent *evt);
    virtual ~HepMCProduct();

    void addHepMCData(HepMC::GenEvent *evt);

    void applyVtxGen(HepMC::FourVector const *vtxShift) { applyVtxGen(*vtxShift); }
    void applyVtxGen(HepMC::FourVector const &vtxShift);

    void boostToLab(TMatrixD const *lorentz, std::string const &type);

    const HepMC::GenEvent &getHepMCData() const;

    const HepMC::GenEvent *GetEvent() const { return evt_; }

    bool isVtxGenApplied() const { return isVtxGenApplied_; }
    bool isVtxBoostApplied() const { return isVtxBoostApplied_; }
    bool isPBoostApplied() const { return isPBoostApplied_; }

    HepMCProduct(HepMCProduct const &orig);
    HepMCProduct &operator=(HepMCProduct const &other);
    HepMCProduct(HepMCProduct &&orig);
    HepMCProduct &operator=(HepMCProduct &&other);
    void swap(HepMCProduct &other);

  private:
    HepMC::GenEvent *evt_;

    bool isVtxGenApplied_;
    bool isVtxBoostApplied_;
    bool isPBoostApplied_;
  };

  // This allows edm::Refs to work with HepMCProduct
  namespace refhelper {
    template <>
    struct FindTrait<edm::HepMCProduct, HepMC::GenParticle> {
      struct Find {
        using first_argument_type = edm::HepMCProduct const &;
        using second_argument_type = int;
        using result_type = HepMC::GenParticle const *;

        result_type operator()(first_argument_type iContainer, second_argument_type iBarCode) {
          return iContainer.getHepMCData().barcode_to_particle(iBarCode);
        }
      };

      typedef Find value;
    };

    template <>
    struct FindTrait<edm::HepMCProduct, HepMC::GenVertex> {
      struct Find {
        using first_argument_type = edm::HepMCProduct const &;
        using second_argument_type = int;
        using result_type = HepMC::GenVertex const *;

        result_type operator()(first_argument_type iContainer, second_argument_type iBarCode) {
          return iContainer.getHepMCData().barcode_to_vertex(iBarCode);
        }
      };

      typedef Find value;
    };
  }  // namespace refhelper
}  // namespace edm

#endif  // SimDataFormats_GeneratorProducts_HepMCProduct_h
