// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemNumberMerger
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  R. Capra
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemNumberMerger.cc,v 1.1 2006/05/17 16:18:58 sunanda Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
TotemNumberMerger :: TotemNumberMerger() {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "Creating TotemNumberMerger";
#endif
}

TotemNumberMerger :: ~TotemNumberMerger() {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "Destruction of TotemNumberMerger";
#endif
}

//
// member functions
//

unsigned long TotemNumberMerger :: Merge(unsigned long value1, 
					 unsigned long value2) const {

  unsigned long c(value1+value2);
  unsigned long result(((c*(c+1))>>1)+value1);

#ifdef SCRIVI
  LogDebug("ForwardSim") << "Merge(value1=" << value1
			 << ", value2=" << value2 << ")=" << result;
  
  unsigned long invValue1, invValue2;
  Split(result, invValue1, invValue2);
  
  assert(invValue1==value1);
  assert(invValue2==value2);                                  
#endif

  return result;
}

unsigned long TotemNumberMerger :: Merge(unsigned long value1, 
					 unsigned long value2, 
					 unsigned long value3) const {
  return Merge(Merge(value1, value2), value3);
}

unsigned long TotemNumberMerger :: Merge(unsigned long value1, 
					 unsigned long value2, 
					 unsigned long value3, 
					 unsigned long value4) const {
  return Merge(Merge(value1, value2), Merge(value3, value4));
}

void TotemNumberMerger :: Split(unsigned long source, unsigned long &value1, 
				unsigned long &value2) const {
  unsigned long c(static_cast<unsigned long>(floor(sqrt(1.+8.*static_cast<float>(source))*0.5-0.5)));
 
  value1 = source-((c*(c+1))>>1);
  value2 = c - value1;
 
#ifdef SCRIVI
  LogDebug("ForwardSim") << "source=" << source << ", c=" << c
			 << ", value1=" << value1 << ", value2=" << value2;
#endif
}

void TotemNumberMerger :: Split(unsigned long source, unsigned long &value1, 
				unsigned long &value2,
				unsigned long &value3) const {
  unsigned long mix12;
 
  Split(source, mix12, value3);
  Split(mix12, value1, value2);
}

void TotemNumberMerger :: Split(unsigned long source,  unsigned long &value1, 
				unsigned long &value2, unsigned long &value3,
				unsigned long &value4) const {
  unsigned long mix12, mix34;
 
  Split(source, mix12, mix34);
  Split(mix12, value1, value2);
  Split(mix34, value3, value4);
}
