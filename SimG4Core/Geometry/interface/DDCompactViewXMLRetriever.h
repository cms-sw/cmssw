#ifndef SimG4Core_DDCompactViewXMLRetriever_H
#define SimG4Core_DDCompactViewXMLRetriever_H

#include "FWCore/CoreFramework/interface/ESProducer.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "SimG4Core/Geometry/interface/PerfectGeometryRecord.h"

#include <string>

class DDCompactViewXMLRetriever : public edm::eventsetup::ESProducer, 
                                  public edm::eventsetup::EventSetupRecordIntervalFinder
{
public:
    DDCompactViewXMLRetriever(std::string & GeomConfig);
    virtual ~DDCompactViewXMLRetriever(); 
    const DDCompactView * produce(const PerfectGeometryRecord &);
protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
				const edm::Timestamp &,edm::ValidityInterval &);
private:
    DDCompactViewXMLRetriever(const DDCompactViewXMLRetriever &);
    const DDCompactViewXMLRetriever & operator=(const DDCompactViewXMLRetriever &);
};


#endif
