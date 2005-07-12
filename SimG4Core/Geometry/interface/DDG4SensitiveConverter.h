#ifndef SimG4Core_DDG4SensitiveConverter_h
#define SimG4Core_DDG4SensitiveConverter_h

#include "SimG4Core/Notification/interface/DDG4DispContainer.h"
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"

#include <iostream>
#include <vector>
#include <string>

class DDG4SensitiveConverter
{
public:    
    DDG4SensitiveConverter();
    virtual ~DDG4SensitiveConverter();
    void upDate(const DDG4DispContainer & ddg4s);
private:
    string getString(const std::string &, const DDLogicalPart *);
};

#endif
