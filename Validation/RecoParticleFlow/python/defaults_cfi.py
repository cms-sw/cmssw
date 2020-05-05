#response config

ptbins = [
    15, 21, 28, 37, 49, 64, 84, 114, 153, 196, 272, 330, 395, 468, 548, 686, 846, 1032, 1248, 1588, 2000, 2500, 3103, 3832, 4713, 5777, 7000
    #10,24,32,43,56,74,97,133,174,245,300,362,430,507,592,686,846,1032,1248,1588,2000,2500,3000,4000,6000
]
etabins = [0.0, 0.5, 1.3, 2.1, 2.5, 3.0, 5.0]

def response_distribution_name(iptbin, ietabin):
    #convert 0.5 -> "05"
    eta_string = "{0:.1f}".format(etabins[ietabin+1]).replace(".", "")
    return "reso_dist_{0:.0f}_{1:.0f}_eta{2}".format(ptbins[iptbin], ptbins[iptbin+1], eta_string)

def genjet_distribution_name(ietabin):
    eta_string = "{0:.1f}".format(etabins[ietabin+1]).replace(".", "")
    return "genjet_pt_eta{0}".format(eta_string)

jetResponseDir = 'ParticleFlow/JetResponse/'
genjetDir = 'ParticleFlow/GenJets/'

#offset config

etaBinsOffset = [-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853, -2.65,
   -2.5, -2.322, -2.172, -2.043, -1.93, -1.83, -1.74, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957,
   -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0,
   0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.879, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479,
   1.566, 1.653, 1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013,
   4.191, 4.363, 4.538, 4.716, 4.889, 5.191]

muLowOffset = 0
muHighOffset = 100

npvLowOffset = 0
npvHighOffset = 100

eBinsOffset = 1000
eLowOffset = 0
eHighOffset = 1000

'''
Defined in PFCandidate.cc
ParticleTypes (use absolute value)
211 h,       // charged hadron
11  e,       // electron
13  mu,      // muon
22  gamma,   // photon
130 h0,      // neutral hadron
1   h_HF,        // HF tower identified as a hadron
2   egamma_HF    // HF tower identified as an EM particle
'''

#map cmssw reco::PFCandidate::pdgId() particle type to our type
candidateDict = {211:"chm", 11:"lep", 13:"lep", 22:"ne", 130:"nh", 1:"hfh", 2:"hfe"}
#our particle types (note chu = unmatched charged hadrons)
candidateType = ["chm", "chu", "nh", "ne", "hfh", "hfe", "lep"]

# map reco::PFCandidate::pdgId() to particle type for PFCand
pdgIDDict = {211: "chargedHadron", 11:"electron", 13:"muon", 22:"photon", 130:"neutralHadron", 1:"HF_hadron", 2: "HF_EM_particle"}

offsetPlotBaseName = 'p_offset_eta'
offsetDir = 'ParticleFlow/Offset/'
offsetR = 0.4

offsetVariableType = ["npv", "mu"]

def offset_name( var, ivar, itype ) :
    return "{0}_{1}{2}_{3}".format( offsetPlotBaseName, var, ivar, itype )
