#ifndef TrackingTools_TrackAssociator_DetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_DetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     DetIdAssociatorMaker
//
/**\class DetIdAssociatorMaker DetIdAssociatorMaker.h "DetIdAssociatorMaker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 14:52:58 GMT
//

// system include files

// user include files

// forward declarations
class DetIdAssociator;
class DetIdAssociatorRecord;

class DetIdAssociatorMaker {
public:
  DetIdAssociatorMaker() = default;
  DetIdAssociatorMaker(const DetIdAssociatorMaker&) = delete;
  const DetIdAssociatorMaker& operator=(const DetIdAssociatorMaker&) = delete;
  virtual ~DetIdAssociatorMaker() = default;

  // ---------- const member functions ---------------------
  virtual std::unique_ptr<DetIdAssociator> make(const DetIdAssociatorRecord&) const = 0;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

private:
  // ---------- member data --------------------------------
};

#endif
