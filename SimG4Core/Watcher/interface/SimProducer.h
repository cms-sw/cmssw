#ifndef Watcher_SimProducer_h
#define Watcher_SimProducer_h
// -*- C++ -*-
//
// Package:     Watcher
// Class  :     SimProducer
// 
/**\class SimProducer SimProducer.h SimG4Core/Watcher/interface/SimProducer.h

 Description: a SimWatcher which puts data into the edm::Event

 Usage:
    <usage>

*/
//
// Original Author:  Chris D. Jones
//         Created:  Mon Nov 28 16:02:21 EST 2005
// $Id: SimProducer.h,v 1.2 2010/03/26 22:55:39 sunanda Exp $
//

// system include files
#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "FWCore/Framework/interface/EDProducer.h"

// forward declarations
namespace simproducer {
   class ProductInfoBase {
      public:
	 ProductInfoBase(const std::string& iInstanceName):
	    m_instanceName(iInstanceName) {}

	 virtual ~ProductInfoBase() {}

	 const std::string& instanceName() const {
	    return m_instanceName; }

	 virtual void registerProduct(edm::EDProducer*) const = 0;
      private:
	 std::string m_instanceName;
   };

   template<class T>
   class ProductInfo : public ProductInfoBase {
      public:
	 ProductInfo(const std::string& iInstanceName) :
	    ProductInfoBase(iInstanceName) {}

	 void registerProduct(edm::EDProducer* iProd) const {
	    (*iProd). template produces<T>(this->instanceName());
	 }
   };
}

class SimProducer : public SimWatcher
{

   public:
      SimProducer() {}
      //virtual ~SimProducer();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void produce(edm::Event&, const edm::EventSetup&) = 0;
      
      void registerProducts(edm::EDProducer& iProd) {
	 std::for_each(m_info.begin(), m_info.end(),
		       boost::bind(&simproducer::ProductInfoBase::registerProduct,_1, &iProd));
      }
   protected:
      template<class T>
      void produces() {
	 produces<T>("");
      }

      template<class T>
      void produces(const std::string& instanceName) {
	 m_info.push_back( 
			  boost::shared_ptr<simproducer::ProductInfo<T> >(new simproducer::ProductInfo<T>(instanceName) ));
      }

   private:
      SimProducer(const SimProducer&); // stop default

      const SimProducer& operator=(const SimProducer&); // stop default

      // ---------- member data --------------------------------
      std::vector<boost::shared_ptr< simproducer::ProductInfoBase> > m_info;
};


#endif
