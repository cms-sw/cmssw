from ROOT import TCut

ok_eta = TCut("TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.14")
ok_pt = TCut("pt > 20.")

ok_dphi1 = TCut("dphi_pad_odd < 10.")
ok_dphi2 = TCut("dphi_pad_even < 10.")


ok_sh1 = TCut("(has_csc_sh&1) > 0")
ok_sh2 = TCut("(has_csc_sh&2) > 0")
ok_st1 = TCut("(has_csc_strips&1) > 0")
ok_st2 = TCut("(has_csc_strips&2) > 0")
ok_w1 = TCut("(has_csc_wires&1) > 0")
ok_w2 = TCut("(has_csc_wires&2) > 0")
ok_digi1 = TCut("%s && %s" %(ok_st1.GetTitle(),ok_w1.GetTitle()))
ok_digi2 = TCut("%s && %s" %(ok_st2.GetTitle(),ok_w2.GetTitle()))
ok_lct1 = TCut("(has_lct&1) > 0")
ok_lct2 = TCut("(has_lct&2) > 0")
ok_lcths1 = TCut("%s && hs_lct_odd > 4 && hs_lct_odd < 125" %(ok_lct1.GetTitle()))
ok_lcths2 = TCut("%s && hs_lct_odd > 4 && hs_lct_odd < 125" %(ok_lct2.GetTitle()))

ok_gsh1 = TCut("(has_gem_sh&1) > 0")
ok_gsh2 = TCut("(has_gem_sh&2) > 0")
ok_g2sh1 = TCut("(has_gem_sh2&1) > 0")
ok_g2sh2 = TCut("(has_gem_sh2&2) > 0")
ok_gdg1 = TCut("(has_gem_dg&1) > 0")
ok_gdg2 = TCut("(has_gem_dg&2) > 0")
ok_pad1 = TCut("(has_gem_pad&1) > 0")
ok_pad2 = TCut("(has_gem_pad&2) > 0")

ok_pad1_lct1 = TCut("%s && %s" %(ok_pad1.GetTitle(),ok_lct1.GetTitle()))
ok_pad2_lct2 = TCut("%s && %s" %(ok_pad2.GetTitle(),ok_lct2.GetTitle()))

ok_pad1_dphi1 = TCut("%s && %s" %(ok_pad1.GetTitle(),ok_dphi1.GetTitle()))
ok_pad2_dphi2 = TCut("%s && %s" %(ok_pad2.GetTitle(),ok_dphi2.GetTitle()))

ok_lct1_eta = TCut("%s && %s" %(ok_eta.GetTitle(),ok_lct1.GetTitle()))
ok_lct2_eta = TCut("%s && %s" %(ok_eta.GetTitle(),ok_lct2.GetTitle()))



ok_pad1_lct1_eta = TCut("%s && %s && %s" %(ok_pad1.GetTitle(),ok_lct1.GetTitle(),ok_eta.GetTitle()))
ok_pad2_lct2_eta = TCut("%s && %s && %s" %(ok_pad2.GetTitle(),ok_lct2.GetTitle(),ok_eta.GetTitle()))

ok_gsh1_lct1_eta = TCut("%s && %s && %s" %(ok_gsh1.GetTitle(),ok_lct1.GetTitle(),ok_eta.GetTitle()))
ok_gsh2_lct2_eta = TCut("%s && %s && %s" %(ok_gsh2.GetTitle(),ok_lct2.GetTitle(),ok_eta.GetTitle()))

ok_gsh1_eta = TCut("%s && %s" %(ok_gsh1.GetTitle(),ok_eta.GetTitle()))
ok_gsh2_eta = TCut("%s && %s" %(ok_gsh2.GetTitle(),ok_eta.GetTitle()))

ok_gdg1_eta = TCut("%s && %s" %(ok_gdg1.GetTitle(),ok_eta.GetTitle()))
ok_gdg2_eta = TCut("%s && %s" %(ok_gdg2.GetTitle(),ok_eta.GetTitle()))

ok_2pad1 = TCut("(has_gem_pad2&1) > 0")
ok_2pad2 = TCut("(has_gem_pad2&2) > 0")

ok_pad1_overlap = TCut("%s || (%s && %s)" %(ok_pad1.GetTitle(),ok_lct2.GetTitle(),ok_pad2.GetTitle()))
ok_pad2_overlap = TCut("%s || (%s && %s)" %(ok_pad2.GetTitle(),ok_lct1.GetTitle(),ok_pad1.GetTitle()))

ok_copad1 = TCut("(has_gem_copad&1) > 0")
ok_copad2 = TCut("(has_gem_copad&2) > 0")

ok_Qp = TCut("charge > 0")
ok_Qn = TCut("charge < 0")

ok_lct1_eta_Qn = TCut("%s && %s && %s" %(ok_lct1.GetTitle(),ok_eta.GetTitle(),ok_Qn.GetTitle()))
ok_lct2_eta_Qn = TCut("%s && %s && %s" %(ok_lct2.GetTitle(),ok_eta.GetTitle(),ok_Qn.GetTitle()))

ok_lct1_eta_Qp = TCut("%s && %s && %s" %(ok_lct1.GetTitle(),ok_eta.GetTitle(),ok_Qp.GetTitle()))
ok_lct2_eta_Qp = TCut("%s && %s && %s" %(ok_lct2.GetTitle(),ok_eta.GetTitle(),ok_Qp.GetTitle()))



Ep = TCut("endcap > 0")
En = TCut("endcap < 0")

