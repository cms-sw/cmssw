#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorMakerBase_h
#define SimG4Core_SensitiveDetector_SensitiveDetectorMakerBase_h
// -*- C++ -*-
//
// Package:     SensitiveDetector
// Class  :     SensitiveDetectorMakerBase
// 
// Original Author:  
//         Created:  Mon Nov 14 11:50:24 EST 2005
//

// system include files
#include <string>
#include <memory>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
// forward declarations
class SimActivityRegistry;
class DDCompactView;
class SimTrackManager;

namespace edm{
  class ParameterSet;
}

class SensitiveDetectorMakerBase
{

public:
  SensitiveDetectorMakerBase(){}
  virtual ~SensitiveDetectorMakerBase(){}

  // ---------- const member functions ---------------------
  virtual void make(const std::string& iname,
		    const DDCompactView& cpv,
		    const SensitiveDetectorCatalog& clg,
		    const edm::ParameterSet& p,
		    const SimTrackManager* man,
		    SimActivityRegistry& reg,
		    std::auto_ptr<SensitiveTkDetector>& oTK,
		    std::auto_ptr<SensitiveCaloDetector>& oCalo) const =0;
      
protected:
  //used to identify which type of Sensitive Detector we have
  void convertTo( SensitiveTkDetector* iFrom, 
		  std::auto_ptr<SensitiveTkDetector>& oTo,
		  std::auto_ptr<SensitiveCaloDetector>) const{
    oTo = std::auto_ptr<SensitiveTkDetector>(iFrom);
  }

  void convertTo( SensitiveCaloDetector* iFrom,
		  std::auto_ptr<SensitiveTkDetector>,
		  std::auto_ptr<SensitiveCaloDetector>& oTo) const{
    oTo = std::auto_ptr<SensitiveCaloDetector>(iFrom);
  }

private:
  SensitiveDetectorMakerBase(const SensitiveDetectorMakerBase&) = delete;
  const SensitiveDetectorMakerBase& operator=(const SensitiveDetectorMakerBase&) = delete;
};


#endif
