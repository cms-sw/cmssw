#ifndef SensitiveDetector_SensitiveDetectorMaker_h
#define SensitiveDetector_SensitiveDetectorMaker_h
// -*- C++ -*-
//
// Package:     SensitiveDetector
// Class  :     SensitiveDetectorMaker
// 
/**\class SensitiveDetectorMaker SensitiveDetectorMaker.h SimG4Core/SensitiveDetector/interface/SensitiveDetectorMaker.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Nov 14 11:56:05 EST 2005
// $Id: SensitiveDetectorMaker.h,v 1.3 2007/05/08 23:11:53 sunanda Exp $
//

// system include files
#include <memory>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"

// forward declarations

template<class T>
class SensitiveDetectorMaker : public SensitiveDetectorMakerBase
{

   public:
     SensitiveDetectorMaker(){}
     //virtual ~SensitiveDetectorMaker();

      // ---------- const member functions ---------------------
      virtual void make(const std::string& iname,
			const DDCompactView& cpv,
			SensitiveDetectorCatalog& clg,
			const edm::ParameterSet& p,
			const SimTrackManager* m,
			SimActivityRegistry& reg,
			std::auto_ptr<SensitiveTkDetector>& oTK,
			std::auto_ptr<SensitiveCaloDetector>& oCalo) const
      {
	std::auto_ptr<T> returnValue(new T(iname, cpv, clg, p, m));
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());

	this->convertTo(returnValue.get(), oTK,oCalo);
	//ownership was passed in the previous function
	returnValue.release();
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      SensitiveDetectorMaker(const SensitiveDetectorMaker&); // stop default

      const SensitiveDetectorMaker& operator=(const SensitiveDetectorMaker&); // stop default

      // ---------- member data --------------------------------

};


#endif
