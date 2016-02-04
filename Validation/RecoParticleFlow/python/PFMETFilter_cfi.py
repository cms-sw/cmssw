import FWCore.ParameterSet.Config as cms

metFilter = cms.EDFilter("PFMETFilter",
#  Collections = cms.vstring("genMetTrue", "pfMet-genMetTrue", "pfMet-genMetTrue"),
#  Variables   = cms.vstring("et",         "et",               "phi"),
#  Mins = cms.vdouble(30.0,200.0,2.0),
#  Maxs = cms.vdouble(-1.0,-200.0,-2.0),
#  DoMin = cms.vint32(1,1,1),
#  DoMax = cms.vint32(0,1,1),
  Collections = cms.vstring("pfMet"),
  Variables   = cms.vstring("DeltaMEXcut"),
  Mins = cms.vdouble(-1.0),
  Maxs = cms.vdouble(-1.0),
  DoMin = cms.vint32(0),
  DoMax = cms.vint32(0),
  verbose = cms.bool(False),
  # 1 = true
  # I do not use vbool because the getParameter function for vbool
  # is not implemented in FWCore/ParameterSet/src/ParameterSet.cc

  # parameters for the cut: sqrt(DeltaMEX**2+DeltaMEY**2)>DeltaMEXsigma*sigma,
  # with sigma=sigma_a+sigma_b*sqrt(SET)+sigma_c*SET
  TrueMET = cms.string("genMetTrue"),
  DeltaMEXsigma = cms.double(7.0),
  sigma_a = cms.double(0.0),
  sigma_b = cms.double(0.5),
  sigma_c = cms.double(0.006)
                         
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
