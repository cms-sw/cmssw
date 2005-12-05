#ifndef SimG4Core_XMLMagneticFieldESSource_H
#define SimG4Core_XMLMagneticFieldESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <memory>
#include <string>

class MagneticField;

class XMLMagneticFieldESSource : public edm::ESProducer, 
                                 public edm::EventSetupRecordIntervalFinder
{
public:
    XMLMagneticFieldESSource(const edm::ParameterSet & p);
    virtual ~XMLMagneticFieldESSource(); 
    std::auto_ptr<DDCompactView> produce(const IdealMagneticFieldRecord &);
protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                const edm::IOVSyncValue &,edm::ValidityInterval &);
private:
    XMLMagneticFieldESSource(const XMLMagneticFieldESSource &);
    const XMLMagneticFieldESSource & operator=(const XMLMagneticFieldESSource &);
    std::string rootNodeName_;
};


#endif

