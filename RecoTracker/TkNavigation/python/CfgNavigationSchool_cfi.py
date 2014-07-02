import FWCore.ParameterSet.Config as cms

CfgNavigationSchoolESProducer = cms.ESProducer("CfgNavigationSchoolESProducer",
                                               ComponentName = cms.string('CfgNavigationSchool'),
                                               SimpleMagneticField = cms.string(''),
#                                               SimpleMagneticField = cms.string('ParabolicMf'),
                                               )

_defaultPSetWithIn=cms.PSet(IN = cms.vstring(''),OUT = cms.vstring(''))
_defaultPSetInverseRelation=cms.PSet(OUT = cms.vstring(''))
_defaultPSet=_defaultPSetWithIn;
parts={}
parts["TIB%d"]=4
parts["TOB%d"]=6
parts["TEC%d_pos"]=9
parts["TEC%d_neg"]=9
parts["TID%d_pos"]=3
parts["TID%d_neg"]=3
parts["PXB%d"]=3
parts["PXF%d_pos"]=2
parts["PXF%d_neg"]=2

import copy
for p in parts.keys():
    for i in range(1,parts[p]+1):
        setattr(CfgNavigationSchoolESProducer,p%(i,),copy.copy(_defaultPSet))
    
