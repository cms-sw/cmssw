#ifndef Ring_h
#define Ring_h

/** \class Ring
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Oct. 14, 2005  
 *
 ************************************************************/

#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <utility>

#include "DataFormats/DetId/interface/DetId.h"

class Ring {
  
 public:

  typedef std::multimap<double,DetId> DetIdMap;
  typedef DetIdMap::iterator iterator;
  typedef DetIdMap::const_iterator const_iterator;
  typedef std::pair<iterator,iterator> IteratorRange;
  typedef std::pair<const_iterator,const_iterator> ConstIteratorRange;

  enum type {
    TIBRing,
    TOBRing,
    TIDRing,
    TECRing,
    PXBRing,
    PXFRing,
    Unspecified
  };

  Ring() : initialized_(false), 
	   rmin_(0.), 
	   rmax_(0.), 
	   zmin_(0.), 
	   zmax_(0.), 
	   type_(Unspecified), 
	   index_(0) {}

  Ring(type input) : initialized_(false), 
		     rmin_(0.), 
		     rmax_(0.), 
		     zmin_(0.), 
		     zmax_(0.), 
		     type_(input), 
		     index_(0) {}

  Ring(unsigned int index, 
       float rmin, 
       float rmax, 
       float zmin, 
       float zmax, 
       unsigned int type) : initialized_(true), 
			    rmin_(rmin), 
			    rmax_(rmax), 
			    zmin_(zmin), 
			    zmax_(zmax), 
			    index_(index) {
    if ( type == 0 ) {
      type_ = TIBRing;
    } else if ( type == 1 ) {
      type_ = TOBRing;
    } else if ( type == 2 ) {
      type_ = TIDRing;
    } else if ( type == 3 ) {
      type_ = TECRing;
    } else if ( type == 4 ) {
      type_ = PXBRing;
    } else if ( type == 5 ) {
      type_ = PXFRing;
    } else {
      type_ = Unspecified;
    } 
  }
  
  Ring(Ring* input) : detids_(input->getDetIdMap()), 
		      initialized_(input->isInitialized()), 
		      rmin_(input->getrmin()), 
		      rmax_(input->getrmax()), 
		      zmin_(input->getzmin()), 
		      zmax_(input->getzmax()), 
		      type_(input->getType()), 
		      index_(input->getindex()) {}

  ~Ring() {}
  
  inline void addId(double phi, DetId id) { detids_.insert(std::make_pair(phi,id)); }

  inline int getNumDetIds() const { return detids_.size(); }
  
  inline bool containsDetId(DetId id) const {
    for ( const_iterator ring = detids_.begin(); ring != detids_.end(); ++ring ) {
      if ( id == ring->second ) {
	return true;
      }
    }
    return false;
  }

  inline const_iterator begin() const { return detids_.begin(); }
  inline const_iterator end()  const  { return detids_.end();   }

  inline iterator begin() { return detids_.begin(); }
  inline iterator end()  { return detids_.end();   }

  inline bool isInitialized() const { return initialized_; }

  inline float getrmin() const { if ( !isInitialized() ) notInitializedMsg(); return rmin_; }
  inline float getrmax() const { if ( !isInitialized() ) notInitializedMsg(); return rmax_; }
  inline float getzmin() const { if ( !isInitialized() ) notInitializedMsg(); return zmin_; }
  inline float getzmax() const { if ( !isInitialized() ) notInitializedMsg(); return zmax_; }

  inline void setrmin(float input) { rmin_ = input; }
  inline void setrmax(float input) { rmax_ = input; }
  inline void setzmin(float input) { zmin_ = input; }
  inline void setzmax(float input) { zmax_ = input; }
  
  inline void setInitialized(bool input) { initialized_ = input; }
  
  inline void initialize(float rmin, float rmax, float zmin, float zmax) { 
    rmin_ = rmin; 
    rmax_ = rmax; 
    zmin_ = zmin; 
    zmax_ = zmax; 
    initialized_ = true; }

  inline void notInitializedMsg() const { 
    std::cout << "[Ring] does not have initialized values for r_min, r_max, z_min, z_max! Using default value of 0. !" << std::endl; }

  inline void setType(type input) { type_ = input; }

  inline type getType() const { return type_; }

  inline DetId getFirst() const { return detids_.begin()->second; }
  
  inline void setindex(unsigned int input) { index_ = input; }

  inline unsigned int getindex() const { return index_; }

  inline DetIdMap getDetIdMap() const { return detids_; }
  
  /// equality
  int operator==(const Ring& ring) const { return index_==ring.getindex(); }
  /// inequality
  int operator!=(const Ring& ring) const { return index_!=ring.getindex(); }
  /// comparison
  int operator<(const Ring& ring) const { return index_<ring.getindex(); }

  inline std::string print() const {
    std::ostringstream stream;
    stream << "Ring: " << index_
	   << " rmin: " << rmin_
	   << " rmax: " << rmax_
	   << " zmin: " << zmin_
	   << " zmax: " << zmax_
	   << " number of DetUnits: " << detids_.size();
    return stream.str();
  }

  inline std::string dump() const {
    std::ostringstream stream;
    stream << "### Ring with index: " << index_ << " ###" << std::endl;
    stream << index_
	   << " " << rmin_
	   << " " << rmax_
	   << " " << zmin_
	   << " " << zmax_ 
	   << " " << type_ << std::endl;
    stream << detids_.size() << std::endl;
    for ( const_iterator entry = detids_.begin(); entry != detids_.end(); ++entry ) {
      stream << entry->first << " " << entry->second.rawId() << std::endl;
    }
    return stream.str();
  }

  inline const_iterator lower_bound(double phi) const { return detids_.lower_bound(phi); }
  inline const_iterator upper_bound(double phi) const { return detids_.upper_bound(phi); }

 private:
  
  DetIdMap detids_;
  
  bool initialized_;

  float rmin_;
  float rmax_;
  float zmin_;
  float zmax_;

  type type_;

  unsigned int index_;

};

#endif
