from ROOT import TCut

#_______________________________________________________________________________
def AND(cut1,cut2):
    return TCut("(%s) && (%s)"%(cut1.GetTitle(),cut2.GetTitle()))

#_______________________________________________________________________________
def OR(cut1,cut2):
    return TCut("(%s) || (%s)"%(cut1.GetTitle(),cut2.GetTitle()))

#_______________________________________________________________________________
ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.14")
ok_pt = TCut("pt > 20.")

## CSC simhits & digis
ok_sh1 = TCut("(has_csc_sh&1) > 0")
ok_sh2 = TCut("(has_csc_sh&2) > 0")
ok_st1 = TCut("(has_csc_strips&1) > 0")
ok_st2 = TCut("(has_csc_strips&2) > 0")
ok_w1 = TCut("(has_csc_wires&1) > 0")
ok_w2 = TCut("(has_csc_wires&2) > 0")
ok_digi1 = AND(ok_st1,ok_w1)
ok_digi2 = AND(ok_st2,ok_w2)

## CSC stub
ok_lct1 = TCut("(has_lct&1) > 0")
ok_lct2 = TCut("(has_lct&2) > 0")
ok_alct1 = TCut("(has_alct&1) > 0")
ok_alct2 = TCut("(has_alct&2) > 0")
ok_clct1 = TCut("(has_clct&1) > 0")
ok_clct2 = TCut("(has_clct&2) > 0")
ok_lct_hs_min = TCut("hs_lct_odd > 4")
ok_lct_hs_max = TCut("hs_lct_odd < 125")
ok_lct_hs = AND(ok_lct_hs_min,ok_lct_hs_max)
ok_lcths1 = AND(ok_lct1,ok_lct_hs)
ok_lcths2 = AND(ok_lct2,ok_lct_hs)

## GEM simhit
ok_gsh1 = TCut("(has_gem_sh&1) > 0")
ok_gsh2 = TCut("(has_gem_sh&2) > 0")
ok_g2sh1 = TCut("(has_gem_sh2&1) > 0")
ok_g2sh2 = TCut("(has_gem_sh2&2) > 0")


## GEM digi
ok_gdg1 = TCut("(has_gem_dg&1) > 0")
ok_gdg2 = TCut("(has_gem_dg&2) > 0")
ok_pad1 = TCut("(has_gem_pad&1) > 0")
ok_pad2 = TCut("(has_gem_pad&2) > 0")

ok_dphi1 = TCut("dphi_pad_odd < 10.")
ok_dphi2 = TCut("dphi_pad_even < 10.")

ok_pad1_lct1 = AND(ok_pad1,ok_lct1)
ok_pad2_lct2 = AND(ok_pad2,ok_lct2)

ok_pad1_dphi1 = AND(ok_pad1,ok_dphi1)
ok_pad2_dphi2 = AND(ok_pad2,ok_dphi2)

ok_lct1_eta = AND(ok_eta,ok_lct1)
ok_lct2_eta = AND(ok_eta,ok_lct2)

ok_pad1_lct1_eta = AND(ok_pad1,AND(ok_lct1,ok_eta))
ok_pad2_lct2_eta = AND(ok_pad2,AND(ok_lct2,ok_eta))

ok_gsh1_lct1_eta = AND(ok_gsh1,AND(ok_lct1,ok_eta))
ok_gsh2_lct2_eta = AND(ok_gsh2,AND(ok_lct2,ok_eta))

ok_gsh1_eta = AND(ok_gsh1,ok_eta)
ok_gsh2_eta = AND(ok_gsh2,ok_eta)

ok_gdg1_eta = AND(ok_gdg1,ok_eta)
ok_gdg2_eta = AND(ok_gdg2,ok_eta)

ok_2pad1 = TCut("(has_gem_pad2&1) > 0")
ok_2pad2 = TCut("(has_gem_pad2&2) > 0")

ok_pad1_overlap = OR(ok_pad1,AND(ok_lct2,ok_pad2))
ok_pad2_overlap = OR(ok_pad2,AND(ok_lct1,ok_pad1))

ok_copad1 = TCut("(has_gem_copad&1) > 0")
ok_copad2 = TCut("(has_gem_copad&2) > 0")

ok_Qp = TCut("charge > 0")
ok_Qn = TCut("charge < 0")

ok_lct1_eta_Qn = AND(ok_lct1,AND(ok_eta,ok_Qn))
ok_lct2_eta_Qn = AND(ok_lct2,AND(ok_eta,ok_Qn))

ok_lct1_eta_Qp = AND(ok_lct1,AND(ok_eta,ok_Qp))
ok_lct2_eta_Qp = AND(ok_lct2,AND(ok_eta,ok_Qp))

Ep = TCut("endcap > 0")
En = TCut("endcap < 0")

