/*************************
 *
 *  Date: 2005/08 $
 *  \author J Weng - F. Moortgat'
 */

#include <iostream>
#include <algorithm> // because we use std::swap

//#include "CLHEP/Vector/ThreeVector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

using namespace edm;
using namespace std;

HepMCProduct::HepMCProduct(HepMC::GenEvent* evt) :
    evt_(evt), isVtxGenApplied_(false), isVtxBoostApplied_(false), isPBoostApplied_(false) {
}

HepMCProduct::~HepMCProduct(){

	delete evt_; evt_ = nullptr; isVtxGenApplied_ = false;
	isVtxBoostApplied_ = false;
	isPBoostApplied_ = false;
}


void HepMCProduct::addHepMCData( HepMC::GenEvent  *evt){
  //  evt->print();
  // cout <<sizeof (evt)  <<"  " << sizeof ( HepMC::GenEvent)   << endl;
  evt_ = evt;
  
  // same story about vertex smearing - GenEvent won't know it...
  // in fact, would be better to implement CmsGenEvent...
  
}

void HepMCProduct::applyVtxGen( HepMC::FourVector* vtxShift )
{
	//std::cout<< " applyVtxGen called " << isVtxGenApplied_ << endl;
	//fTimeOffset = 0;
	
   if ( isVtxGenApplied() ) return ;
      
   for ( HepMC::GenEvent::vertex_iterator vt=evt_->vertices_begin();
                                          vt!=evt_->vertices_end(); ++vt )
   {
            
      double x = (*vt)->position().x() + vtxShift->x() ;
      double y = (*vt)->position().y() + vtxShift->y() ;
      double z = (*vt)->position().z() + vtxShift->z() ;
      double t = (*vt)->position().t() + vtxShift->t() ;
      //std::cout << " vertex (x,y,z)= " << x <<" " << y << " " << z << std::endl;
      (*vt)->set_position( HepMC::FourVector(x,y,z,t) ) ;      
   }
      
   isVtxGenApplied_ = true ;
   
   return ;

} 

void HepMCProduct::boostToLab( TMatrixD* lorentz, std::string type ) {

	//std::cout << "from boostToLab:" << std::endl;
	
  
  
	if ( lorentz == 0 ) {

		//std::cout << " lorentz = 0 " << std::endl;
		return;
	}

	//lorentz->Print();

  TMatrixD tmplorentz(*lorentz);
  //tmplorentz.Print();
  
	if ( type == "vertex") {

		if ( isVtxBoostApplied() ) {
			//std::cout << " isVtxBoostApplied true " << std::endl;
			return ;
		}
		
		for ( HepMC::GenEvent::vertex_iterator vt=evt_->vertices_begin();
			  vt!=evt_->vertices_end(); ++vt ) {

			// change basis to lorentz boost definition: (t,x,z,y)
			TMatrixD p4(4,1);
			p4(0,0) = (*vt)->position().t();
			p4(1,0) = (*vt)->position().x();
			p4(2,0) = (*vt)->position().z();
			p4(3,0) = (*vt)->position().y();

			TMatrixD p4lab(4,1);
			p4lab = tmplorentz * p4;
			//std::cout << " vertex lorentz: " << p4lab(1,0) << " " << p4lab(3,0) << " " << p4lab(2,0) << std::endl;
			(*vt)->set_position( HepMC::FourVector(p4lab(1,0),p4lab(3,0),p4lab(2,0), p4lab(0,0) ) ) ;      
		}
      
		isVtxBoostApplied_ = true ;
	}
	else if ( type == "momentum") {
		
		if ( isPBoostApplied() ) {
			//std::cout << " isPBoostApplied true " << std::endl;
			return ;
		}
		
		for ( HepMC::GenEvent::particle_iterator part=evt_->particles_begin();
			  part!=evt_->particles_end(); ++part ) {

			// change basis to lorentz boost definition: (E,Px,Pz,Py)
			TMatrixD p4(4,1);
			p4(0,0) = (*part)->momentum().e();
			p4(1,0) = (*part)->momentum().x();
			p4(2,0) = (*part)->momentum().z();
			p4(3,0) = (*part)->momentum().y();

			TMatrixD p4lab(4,1);
			p4lab = tmplorentz * p4;
			//std::cout << " momentum lorentz: " << p4lab(1,0) << " " << p4lab(3,0) << " " << p4lab(2,0) << std::endl;
			(*part)->set_momentum( HepMC::FourVector(p4lab(1,0),p4lab(3,0),p4lab(2,0),p4lab(0,0) ) ) ;      
		}
      
		isPBoostApplied_ = true ;
	}
	else {
		std::cout << " no type found for boostToLab(std::string), options are vertex or momentum" << std::endl;
	}
		
		 
	return ;
}


const HepMC::GenEvent&  
HepMCProduct::getHepMCData()const   {

  return  * evt_;
}

// copy constructor
HepMCProduct::HepMCProduct(HepMCProduct const& other) :
  evt_(nullptr) {
  
   if (other.evt_) evt_=new HepMC::GenEvent(*other.evt_);
   isVtxGenApplied_ = other.isVtxGenApplied_ ;
   isVtxBoostApplied_ = other.isVtxBoostApplied_;
   isPBoostApplied_ = other.isPBoostApplied_;
   //fTimeOffset = other.fTimeOffset;
}

// swap
void
HepMCProduct::swap(HepMCProduct& other) {
  std::swap(evt_, other.evt_);
  std::swap(isVtxGenApplied_, other.isVtxGenApplied_);
  std::swap(isVtxBoostApplied_, other.isVtxBoostApplied_);
  std::swap(isPBoostApplied_, other.isPBoostApplied_);
  //std::swap(fTimeOffset, other.fTimeOffset);
}

// assignment: use copy/swap idiom for exception safety.
HepMCProduct&
HepMCProduct::operator=(HepMCProduct const& other) {
  HepMCProduct temp(other);
  swap(temp);
  return *this;
} 
