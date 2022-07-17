/*************************
 *
 *  Date: 2005/08 $
 *  \author J Weng - F. Moortgat'
 */

#include <iostream>
#include <algorithm>  // because we use std::swap

#include <HepMC3/GenEvent.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/GenParticle.h>
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

HepMC3Product::HepMC3Product(HepMC3::GenEvent* evt)
    : evt_(evt), isVtxGenApplied_(false), isVtxBoostApplied_(false), isPBoostApplied_(false) {}

HepMC3Product::~HepMC3Product() {
  delete evt_;
  evt_ = nullptr;
  isVtxGenApplied_ = false;
  isVtxBoostApplied_ = false;
  isPBoostApplied_ = false;
}

void HepMC3Product::addHepMCData(HepMC3::GenEvent* evt) {
  //  evt->print();
  // cout <<sizeof (evt)  <<"  " << sizeof ( HepMC::GenEvent)   << endl;
  evt_ = evt;

  // same story about vertex smearing - GenEvent won't know it...
  // in fact, would be better to implement CmsGenEvent...
}

void HepMC3Product::applyVtxGen(HepMC3::FourVector const& vtxShift) {
  //std::cout<< " applyVtxGen called " << isVtxGenApplied_ << endl;
  //fTimeOffset = 0;
  if (isVtxGenApplied())
    return;
  evt_->shift_position_by(vtxShift);
  isVtxGenApplied_ = true;
  return;
}

void HepMC3Product::boostToLab(TMatrixD const* lorentz, std::string const& type) {
  //std::cout << "from boostToLab:" << std::endl;

  if (lorentz == nullptr) {
    //std::cout << " lorentz = 0 " << std::endl;
    return;
  }

  //lorentz->Print();

  TMatrixD tmplorentz(*lorentz);
  //tmplorentz.Print();

  if (type == "vertex") {
    if (isVtxBoostApplied()) {
      //std::cout << " isVtxBoostApplied true " << std::endl;
      return;
    }

    for (const HepMC3::GenVertexPtr& vt : evt_->vertices()) {
      // change basis to lorentz boost definition: (t,x,z,y)
      TMatrixD p4(4, 1);
      p4(0, 0) = vt->position().t();
      p4(1, 0) = vt->position().x();
      p4(2, 0) = vt->position().z();
      p4(3, 0) = vt->position().y();

      TMatrixD p4lab(4, 1);
      p4lab = tmplorentz * p4;
      //std::cout << " vertex lorentz: " << p4lab(1,0) << " " << p4lab(3,0) << " " << p4lab(2,0) << std::endl;
      vt->set_position(HepMC3::FourVector(p4lab(1, 0), p4lab(3, 0), p4lab(2, 0), p4lab(0, 0)));
    }

    isVtxBoostApplied_ = true;
  } else if (type == "momentum") {
    if (isPBoostApplied()) {
      //std::cout << " isPBoostApplied true " << std::endl;
      return;
    }

    for (const HepMC3::GenParticlePtr& part : evt_->particles()) {
      // change basis to lorentz boost definition: (E,Px,Pz,Py)
      TMatrixD p4(4, 1);
      p4(0, 0) = part->momentum().e();
      p4(1, 0) = part->momentum().x();
      p4(2, 0) = part->momentum().z();
      p4(3, 0) = part->momentum().y();

      TMatrixD p4lab(4, 1);
      p4lab = tmplorentz * p4;
      //std::cout << " momentum lorentz: " << p4lab(1,0) << " " << p4lab(3,0) << " " << p4lab(2,0) << std::endl;
      part->set_momentum(HepMC3::FourVector(p4lab(1, 0), p4lab(3, 0), p4lab(2, 0), p4lab(0, 0)));
    }

    isPBoostApplied_ = true;
  } else {
    edm::LogWarning("GeneratorProducts") << "no type found for boostToLab(std::string), options are vertex or momentum";
    //throw edm::Exception(edm::errors::Configuration, "GeneratorProducts")
    //  << "no type found for boostToLab(std::string), options are vertex or momentum \n";
  }

  return;
}

const HepMC3::GenEvent& HepMC3Product::getHepMCData() const { return *evt_; }

// copy constructor
HepMC3Product::HepMC3Product(HepMC3Product const& other) : evt_(nullptr) {
  if (other.evt_)
    evt_ = new HepMC3::GenEvent(*other.evt_);
  isVtxGenApplied_ = other.isVtxGenApplied_;
  isVtxBoostApplied_ = other.isVtxBoostApplied_;
  isPBoostApplied_ = other.isPBoostApplied_;
  //fTimeOffset = other.fTimeOffset;
}

// swap
void HepMC3Product::swap(HepMC3Product& other) {
  std::swap(evt_, other.evt_);
  std::swap(isVtxGenApplied_, other.isVtxGenApplied_);
  std::swap(isVtxBoostApplied_, other.isVtxBoostApplied_);
  std::swap(isPBoostApplied_, other.isPBoostApplied_);
  //std::swap(fTimeOffset, other.fTimeOffset);
}

// assignment: use copy/swap idiom for exception safety.
HepMC3Product& HepMC3Product::operator=(HepMC3Product const& other) {
  HepMC3Product temp(other);
  swap(temp);
  return *this;
}

// move, needed explicitly as we have raw pointer...
HepMC3Product::HepMC3Product(HepMC3Product&& other) : evt_(nullptr) { swap(other); }
HepMC3Product& HepMC3Product::operator=(HepMC3Product&& other) {
  swap(other);
  return *this;
}
