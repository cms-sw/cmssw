#ifndef SimDataFormats_GeneratorProducts_HepMCProduct_h
#define SimDataFormats_GeneratorProducts_HepMCProduct_h

/** \class HepMCProduct
 *
 *  \author Joanna Weng, Filip Moortgat
 */

#include <TMatrixD.h>

#include <cstddef>
#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "DataFormats/Common/interface/Ref.h"

namespace edm {
	class HepMCProduct {
	    public:
		HepMCProduct() :
			evt_(nullptr), isVtxGenApplied_(false),
			isVtxBoostApplied_(false), isPBoostApplied_(false) {}

		explicit HepMCProduct(HepMC::GenEvent *evt);
		virtual ~HepMCProduct();

		void addHepMCData(HepMC::GenEvent *evt);

		void applyVtxGen(HepMC::FourVector *vtxShift);

		void boostToLab(TMatrixD *lorentz, std::string type);

		const HepMC::GenEvent &getHepMCData() const;

		const HepMC::GenEvent *GetEvent() const { return evt_; }

		bool isVtxGenApplied() const { return isVtxGenApplied_; }
		bool isVtxBoostApplied() const { return isVtxBoostApplied_; }
		bool isPBoostApplied() const { return isPBoostApplied_; }

		HepMCProduct(HepMCProduct const &orig);
		HepMCProduct &operator = (HepMCProduct const &other);
		void swap(HepMCProduct &other);

	    private:
		HepMC::GenEvent	*evt_;

		bool	isVtxGenApplied_ ;
		bool	isVtxBoostApplied_;
		bool	isPBoostApplied_;
	
	};

	// This allows edm::Refs to work with HepMCProduct
	namespace refhelper {
		template<> 
		struct FindTrait<edm::HepMCProduct, HepMC::GenParticle> {
			struct Find : public std::binary_function<edm::HepMCProduct const&, int, HepMC::GenParticle const*> {
				typedef Find self;
				self::result_type operator () (self::first_argument_type iContainer,
				                               self::second_argument_type iBarCode)
				{ return iContainer.getHepMCData().barcode_to_particle(iBarCode); }
			};

			typedef Find value;
		};

		template<> 
		struct FindTrait<edm::HepMCProduct, HepMC::GenVertex> {
			struct Find : public std::binary_function<edm::HepMCProduct const&, int, HepMC::GenVertex const*> {
				typedef Find self;

				self::result_type operator () (self::first_argument_type iContainer,
				                               self::second_argument_type iBarCode)
				{ return iContainer.getHepMCData().barcode_to_vertex(iBarCode); }
			};

			typedef Find value;
		};
	} // namespace refhelper
} // namespace edm

#endif // SimDataFormats_GeneratorProducts_HepMCProduct_h
