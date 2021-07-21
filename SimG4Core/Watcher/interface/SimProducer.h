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
//

// system include files
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

// forward declarations
namespace simproducer {
  class ProductInfoBase {
  public:
    ProductInfoBase(const std::string &iInstanceName) : m_instanceName(iInstanceName) {}

    virtual ~ProductInfoBase() {}

    const std::string &instanceName() const { return m_instanceName; }

    virtual void registerProduct(edm::ProducesCollector) const = 0;

  private:
    std::string m_instanceName;
  };

  template <class T>
  class ProductInfo : public ProductInfoBase {
  public:
    ProductInfo(const std::string &iInstanceName) : ProductInfoBase(iInstanceName) {}

    void registerProduct(edm::ProducesCollector producesCollector) const override {
      producesCollector.produces<T>(this->instanceName());
    }
  };
}  // namespace simproducer

class SimProducer : public SimWatcher {
public:
  SimProducer() {}

  virtual void produce(edm::Event &, const edm::EventSetup &) = 0;

  void registerProducts(edm::ProducesCollector producesCollector) {
    std::for_each(m_info.begin(),
                  m_info.end(),
                  [&producesCollector](std::shared_ptr<simproducer::ProductInfoBase> const &ptr) mutable {
                    ptr->registerProduct(producesCollector);
                  });
  }

  SimProducer(const SimProducer &) = delete;
  const SimProducer &operator=(const SimProducer &) = delete;

protected:
  template <class T>
  void produces() {
    produces<T>("");
  }

  template <class T>
  void produces(const std::string &instanceName) {
    m_info.push_back(std::make_shared<simproducer::ProductInfo<T>>(instanceName));
  }

private:
  std::vector<std::shared_ptr<simproducer::ProductInfoBase>> m_info;
};

#endif
