import FWCore.ParameterSet.Config as cms

ecal_notCont_sim = cms.PSet(
    # Values obtained by PM 04/04/2007
    # using CMSSW_2_0_0_pre7
    # Using uncoverted photons E=50
    # Reference region Barrel ietaAbs() > 5 && ietaAbs() < 21 && (iphi() % 20 ) > 5 && (iphi() % 20 ) < 16 
    # Reference region Endcap  fabs(superClusterEta[i]) > 1.7 && fabs(superClusterEta[i] < 2.3)  &&
    #                   (
    #                    (superClusterPhi[i] > -TMath::Pi()/2. + 0.1 && superClusterPhi[i] < TMath::Pi()/2. - 0.1) ||
    #                    (superClusterPhi[i] > TMath::Pi()/2. + 0.1) ||
    #                    (superClusterPhi[i] < -TMath::Pi()/2. - 0.1)
    #                    ) )
    # For Barrel using E_{25}/E_{true}
    # For Endcap using E_{25} + E_{ps} = E_{true} 
    EBs25notContainment = cms.double(0.971),
    EEs25notContainment = cms.double(0.975)
)

