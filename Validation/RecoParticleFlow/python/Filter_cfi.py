import FWCore.ParameterSet.Config as cms

metFilter = cms.EDFilter("PFFilter",
  Collections = cms.vstring("genMetTrue", "pfMet-genMetTrue", "pfMet-genMetTrue"),
  Variables   = cms.vstring("et",         "et",               "phi"),
  Mins = cms.vdouble(30.0,200.0,2.0),
  Maxs = cms.vdouble(-1.0,-200.0,-2.0),
  DoMin = cms.vint32(1,1,1),
  DoMax = cms.vint32(0,1,1)
  # 1 = true
  # I do not use vbool because the getParameter function for vbool
  # is not implemented in FWCore/ParameterSet/src/ParameterSet.cc

# variables can be "et", "eta" or "phi"

#  #: excluded -: kept

# ############|-----------
#            min

# ------------|###########
#            max

# ########|--------|########
#       min       max

# --------|########|--------
#       max       min

)
