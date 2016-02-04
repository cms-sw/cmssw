#ifndef SimDataFormats_Forward_LHCTransportLink_h
#define SimDataFormats_Forward_LHCTransportLink_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     LHCTransportLink
//
/**\class LHCTransportLink LHCTransportLink.h SimG4CMS/Forward/interface/LHCTransportLink.h
 
 Description: correspondence link between barcodes for GenParticle transported by Hector and original ones

 Usage: in SimTrack creation when the Hector beam transport is used
 
*/
//
// Original Author: 
//         Created:  Fri May 29 17:00:00 CEST 2009
// $Id: LHCTransportLink.h,v 1.1 2009/06/10 08:10:26 fabiocos Exp $
//
 
// system include files
#include <iostream>

// user include files

class LHCTransportLink {

 public:
  
  // ---------- Constructor and destructor -----------------
  explicit LHCTransportLink(int & beforeHector, int & afterHector):beforeHector_(beforeHector),afterHector_(afterHector) { };
  LHCTransportLink():beforeHector_(0),afterHector_(0) {}; 
  
  // ---------- Member functions --------------------------- 
  
  void fill(int & afterHector, int & beforeHector);
  int beforeHector() const;
  int afterHector() const;
  void clear();
  
 private: 
  
  // ---------- Private Data members ----------------------- 
  int beforeHector_;
  int afterHector_;
  
};

std::ostream & operator <<(std::ostream & o , const LHCTransportLink & t);

#endif
