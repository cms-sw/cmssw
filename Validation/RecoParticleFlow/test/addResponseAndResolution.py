# This jet pT resolution plotting function creates resolution and response
# histogram files from jet response histograms.
# Run as follows at a PyROOT compatible environment:
# 'python addResponseAndResolution.py path/to/input_file.root'
#
# Output file will have the same name as the input file with the
# "_withResponseAndResolution" appendix added.
#
# The output will be written into the working directory, overwriting the
# existing file with same name if needed.
#
# At the moment the code writes only reso and respo histograms to the output file,
# but it should be extended to copy the contents of the input file to output as well.
#
# Author: juska@cern.ch
# Intro written 28.1.2019
#
# Extended for 'smart gaussian fit' -based resolution and response calculation
# 1 Mar 2019, and renamed the python file and function to reflect this.
# (the name was before very misleading anyway)
#
# For keeping the RMS-based calculations on the side cross-checking our results,
# the function calculates also RMS-based response and resolution histograms,
# and stores them to histograms with the appendix '_rms'.
#
# NOTE: the response given by this function is calculated with corrected jet
# transverse momentum, so it could also be called jec closure.
# We cannot calculate the traditional raw-jet response with the response
# histograms used here, because they are filled using corrected jets, as
# this is how resolution needs to be calculated.
#

import ROOT as r
import sys
import array as ar
import math
import helperFunctions as help

def make_subdirs(tfile, path):
   d = tfile
   for dirname in path.split("/"):
      if len(dirname) > 0:
         d.mkdir(dirname)
         d = d.Get(dirname)
         d.cd()
      
def add_response_and_resolution(sforig, treepath):

   # NOTE if you want to write the output to the input file directory instead,
   # just delete the ".split('/')[-1]" part.
   
   sfout = sforig[0:-5].split('/')[-1] + "_withResponseAndResolution.root"
   forig = r.TFile(sforig,"READ")
   fout = r.TFile(sfout,"RECREATE")

   fout.cd()
   make_subdirs(fout, "%s/JetResponse" % treepath)
   
   # This is needed for smart fit range and may change with samples
   # It's given to the fit_response function in the inner pT loop
   recoptcut = 15.
 
   
   etabins = [[0,0.5],[0.5,1.3],[1.3,2.1],[2.1,2.5],[2.5,3.0]]
   
   etadict = {0.5:"eta05", 1.3:"eta13", 2.1:"eta21", 2.5:"eta25", 3.0:"eta30"}
   
   ptbins = ar.array('d',[10,24,32,43,56,74,97,133,174,245,300,362,430,
                  507,592,686,846,1032,1248,1588,2000,2500,3000,4000,6000])
   nptbins = len(ptbins)-1
   

   
   # Add smart gaussian fit based resolution and response histograms
   for etaidx in range(len(etabins)):
      
      apdx = etadict[etabins[etaidx][1]]
      preso = r.TH1D("preso_%s" % apdx,
          "Jet pT resolution, eta=[{0}, {1}]".format(etabins[etaidx][0],
          etabins[etaidx][1]),nptbins,ptbins )
      response_pt = r.TH1D("presponse_%s" % apdx,
          "Jet pT response, eta=[{0}, {1}]".format(etabins[etaidx][0],
          etabins[etaidx][1]),nptbins,ptbins )

      # get number of gen jets in the pT bin for normalizing the smart fit
      hgenjet = forig.Get("%s/GenJets/genjet_pt_%s" % (treepath,apdx))     

      for idx in range(nptbins):
         elow = ptbins[idx]
         ehigh = ptbins[idx+1]
         h = forig.Get("%s/JetResponse/reso_dist_%i_%i_%s" % (treepath,elow,ehigh,apdx))

         # Find the correct bin and ngenjets
         binhigh = ptbins.index(ehigh)
         ngenjet = hgenjet.GetBinContent(binhigh)
         
         # Don't bother trying to fit empty histos
         if (h.GetEntries() == 0):
            continue

         # Nasty details are hidden here
         response, response_unc, resolution, resolution_unc = \
            help.fit_response(h,ngenjet,elow,recoptcut)
         
         preso.SetBinContent(idx,resolution)
         preso.SetBinError(idx,resolution_unc)
         
         response_pt.SetBinContent(idx,response)
         response_pt.SetBinError(idx,response_unc)
                  
      fout.Write()
   
   # TODO Fix multiple saving error!!
   
   # Add RMS-based resolution and response histograms
   for etaidx in range(len(etabins)):
      
      apdx = etadict[etabins[etaidx][1]]
      preso_rms = r.TH1D("preso_%s_rms" % apdx,
          "Jet pT resolution using RMS, %1.1f < |#eta| < %1.1f" % (etabins[etaidx][0],
          etabins[etaidx][1]),nptbins,ptbins )
      response_pt_rms = r.TH1D("presponse_%s_rms" % apdx,
          "Jet pT response using RMS, %1.1f < |#eta| < %1.1f" % (etabins[etaidx][0],
          etabins[etaidx][1]),nptbins,ptbins )
      

      for idx in range(nptbins):
         elow = ptbins[idx]
         ehigh = ptbins[idx+1]
         h = forig.Get("%s/JetResponse/reso_dist_%i_%i_%s" % (treepath,elow,ehigh,apdx))
         
         std = h.GetStdDev()
         std_error = h.GetStdDevError()
          
         # Scale each bin with mean response
         mean = 1.0
         mean_error = 0.0
         if (h.GetMean()>0):
            mean = h.GetMean()
            mean_error = h.GetMeanError()
         err = 0.0
         if std > 0.0 and mean > 0.0: 
             err = std/mean * math.sqrt(std_error**2 / std**2 + mean_error**2 / mean**2)

         #fill the pt-dependent resolution plot
         preso_rms.SetBinContent(idx, std/mean)
         preso_rms.SetBinError(idx, err)
        
         #fill the pt-dependent response plot with the mean of the response.
         response_pt_rms.SetBinContent(idx, mean)
         err2 = 0.0
         if std > 0.0 and mean > 0.0: 
             err2 = mean * math.sqrt(std_error**2 / std**2 + mean_error**2 / mean**2)
         response_pt_rms.SetBinError(idx, err2)
         
      fout.Write()
   fout.Close()   
      
   
def main():

   # Read input file path from command line argumets
   
   argv = sys.argv
   sforig = argv[1]
   
   treepath = "DQMData/Run 1/Physics/Run summary"
   
   add_response_and_resolution(sforig, treepath)

main()
