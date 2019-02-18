# This jet pT resolution plotting function creates resolution histogram
# files from jet response histograms.
# Run as follows at a PyROOT compatible environment:
# 'python addResponse.py path/to/input_file.root'
#
# Output file will have the same name as the input file with the "_withResponse"
# appendix added.
#
# The output will be written into the working directory, overwriting the
# existing file with same name if needed.
#
# At the moment the code writes only resolution profiles to the output file, but
# should be extended to copy the contents of the input file to output as well.
#
# Author: juska@cern.ch
# Intro written 28.1.2019

import ROOT as r
import sys
import array as ar
import math

def make_subdirs(tfile, path):
   d = tfile
   for dirname in path.split("/"):
      if len(dirname) > 0:
         d.mkdir(dirname)
         d = d.Get(dirname)
         d.cd()
      
def pf_resolution(sforig, treepath):

   # NOTE if you want to write the output to the input file directory instead,
   # just delete the ".split('/')[-1]" part.
   
   sfout = sforig[0:-5].split('/')[-1] + "_withResponse.root"
   forig = r.TFile(sforig,"READ")
   fout = r.TFile(sfout,"RECREATE")
   print sfout
   fout.cd()
   make_subdirs(fout, treepath)
 
   #creso = r.TCanvas("resolution","Jet pT resolution", 600, 600)
   
   etabins = [[0,0.5],[0,1.3],[1.3,2.1],[2.1,2.5],[2.5,3.0]]
   
   etadict = {0.5:"eta05", 1.3:"eta13", 2.1:"eta21", 2.5:"eta25", 3.0:"eta30"}
   
   ptbins = ar.array('d',[10,24,32,43,56,74,97,133,174,245,300,362,430,
                  507,592,686,846,1032,1248,1588,2000,2500,3000,4000,6000])
   nptbins = len(ptbins)-1
   
   
   # Loop responses i.e. dist widths to a TProfile.
   
   for etaidx in range(len(etabins)):
      
      apdx = etadict[etabins[etaidx][1]]
      preso = r.TH1D("preso_%s" % apdx,
          "Jet pT resolution, eta=[{0}, {1}]".format(etabins[etaidx][0], etabins[etaidx][1]),nptbins,ptbins
      )
      response_pt = r.TH1D("presponse_%s" % apdx,
          "Jet pT response, eta=[{0}, {1}]".format(etabins[etaidx][0], etabins[etaidx][1]),nptbins,ptbins
      )
      

      for idx in range(nptbins):
         elow = ptbins[idx]
         ehigh = ptbins[idx+1]
         h = forig.Get("%sreso_dist_%i_%i_%s" % (treepath,elow,ehigh,apdx))
         
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
         preso.SetBinContent(idx, std/mean)
         preso.SetBinError(idx, err)
        
         #fill the pt-dependent response plot with the mean of the response.
         response_pt.SetBinContent(idx, mean)
         err2 = 0.0
         if std > 0.0 and mean > 0.0: 
             err2 = mean * math.sqrt(std_error**2 / std**2 + mean_error**2 / mean**2)
         response_pt.SetBinError(idx, err2)
         
      fout.Write()
   fout.Close()   
      
   
def main():

   # Read input file path from command line argumets
   
   argv = sys.argv
   sforig = argv[1]
   
   treepath = "DQMData/Run 1/Physics/Run summary/JetResponse/"
   
   pf_resolution(sforig, treepath)

main()
