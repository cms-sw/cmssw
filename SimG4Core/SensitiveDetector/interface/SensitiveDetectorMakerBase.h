#ifndef SensitiveDetector_SensitiveDetectorMakerBase_h
#define SensitiveDetector_SensitiveDetectorMakerBase_h
// -*- C++ -*-
//
// Package:     SensitiveDetector
// Class  :     SensitiveDetectorMakerBase
// 
/**\class SensitiveDetectorMakerBase SensitiveDetectorMakerBase.h SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Nov 14 11:50:24 EST 2005
//

// system include files
#include <string>

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
			const SimTrackManager* m,
			SimActivityRegistry& reg,
			std::unique_ptr<SensitiveTkDetector>& oTK,
			std::unique_ptr<SensitiveCaloDetector>& oCalo) const =0;
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   protected:
      //used to identify which type of Sensitive Detector we have
      void convertTo( SensitiveTkDetector* iFrom, 
		      std::unique_ptr<SensitiveTkDetector>& oTo,
		      std::unique_ptr<SensitiveCaloDetector>&) const{
	oTo= std::unique_ptr<SensitiveTkDetector>(iFrom);
      }
      void convertTo( SensitiveCaloDetector* iFrom,
		      std::unique_ptr<SensitiveTkDetector>&,
		      std::unique_ptr<SensitiveCaloDetector>& oTo) const{
	oTo=std::unique_ptr<SensitiveCaloDetector>(iFrom);
      }

   private:
      SensitiveDetectorMakerBase(const SensitiveDetectorMakerBase&); // stop default

      const SensitiveDetectorMakerBase& operator=(const SensitiveDetectorMakerBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
