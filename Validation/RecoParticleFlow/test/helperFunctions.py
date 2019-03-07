# This file contains miscellaneous helper functions for the PF validation
# package.
#
# First developer: juska@cern.ch
# Created: 27 Feb 2019

#
# The 'fit_response'function makes a 'smart gaus fit' to the response histogram
# and by using the number of generator jets as an extra bit of information
# constrains the fit so that the issues with the low-pT bias arising from jet pT
# cuts in the sample are fixed to some extent.
#
# The 'corrected response / jec closure' is also calculated from the smart fit,
# and the method returns the following float variables in the following order:
# 
# jet response, jet resp uncertainty, jet resolution, jet resolution uncertainty 
#

import ROOT as r
 
def fit_response(hreso,ngenjet,elow,recoptcut):
   # Call as: fit_response(histo,123,15)
   
   # Only do plots if needed for debugging
   do_plots = False

   cfit = r.TCanvas("respofit","respofit", 600, 600)
   
   #hreso = drespo.Get("JetResponse/reso_dist_%i_%i_%s" % (elow,ehigh,seta))
   #print type(hreso)

   ## Get ngenjet in the pt bin   
   #hgenj = drespo.Get("GenJets/genjet_pt_%s" % seta)
   
   # Integral integrates from bin to bin. Of course.
   # index() give the bin below when given a bin boundary
   #binhigh = ptbins.index(ehigh)
   #ngenjet = hgenj.GetBinContent(binhigh)

   # Take range by Mikko's advice: -1.5 and + 1.5 * RMS width
   
   rmswidth = hreso.GetStdDev()
   rmsmean = hreso.GetMean()
   
   fitlow = rmsmean-1.5*rmswidth
   
   fitlow = max(recoptcut/elow,fitlow)
   
   #print "fitlow: " + str(fitlow)
   
   fithigh = rmsmean+1.5*rmswidth

   fg = r.TF1("mygaus","gaus",fitlow,fithigh)
   
   fg2 = r.TF1("fg2","TMath::Gaus(x,[0],[1],true)*[2]",fitlow,fithigh)
   binlow = hreso.FindBin(fitlow)
   binhigh = hreso.FindBin(fithigh)
   
   area = hreso.Integral(binlow,binhigh)
   
   #fg.Scale(1./fg.Integral(-10,10))
   
   #print "area before: " + str(area)
   #print "ngenjet: " + str(ngenjet)
   
   hreso.Fit("mygaus", "RQ")
   
   fg2.SetParameter(0,fg.GetParameter(1))
   fg2.SetParameter(1,fg.GetParameter(2))
   
   # Here the fit is forced to take the area of ngenjets.
   # The area is further normalized for the response histogram x-axis lenght
   # (3) and number of bins (100)
   fg2.FixParameter(2,ngenjet*3./100.)#/np.sqrt(2.))
   
   hreso.Fit("fg2","RQ")
   
   fitlow = fg2.GetParameter(0)-1.5*fg2.GetParameter(1)
   fitlow = max(15./elow,fitlow)
   
   #print "fitlow: " + str(fitlow)
   
   fithigh = fg2.GetParameter(0)+1.5*fg2.GetParameter(1)
      
   fg2.SetRange(fitlow,fithigh)
   
   hreso.Fit("fg2","RQ")
   
   fg.SetRange(0,3)
   fg2.SetRange(0,3)
   fg.SetLineWidth(2)
   fg2.SetLineColor(r.kGreen+2)
   
   hreso.GetXaxis().SetRangeUser(0,2)
   
   
   # Save plots to a subdirectory if asked 
   if do_plots and hreso.GetEntries()>0:
      hreso.Draw("ehist")
      fg.Draw("same")
      fg2.Draw("same")
      cfit.SaveAs("res/respo_fit/respo_smartfit_%04d_%i_%s.pdf" %
            (elow,ehigh,seta))
            
   resp, resp_err = fg2.GetParameter(0), fg2.GetParError(0)
   reso, reso_err = fg2.GetParameter(1), fg2.GetParError(1)
   
   # Avoid division by zero error:
   if (resp == 0):
      return resp,resp_err,0,reso_err
   
   # Resolution is scaled with response
   return resp, resp_err, reso/resp, reso_err
   

# Calculate resolution uncertainty   
def get_resp_unc(width, width_err, mean, mean_err):
   if 0==width or 0==mean:
      return 0
   return r.TMath.Sqrt(width_err**2/width**2+mean_err**2/mean**2)*width


