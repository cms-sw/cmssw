- The analyzer output files are stored in gem_vadim  (for default and gem configuration)
- To generate the plots need to start root 
               * root -l 
               * .L drawplot_gmtrt()  (It has some basic functions used in frankenstein)
               * drawplot_frankenstein.C()
- This generates a bunch of rate histograms depending on the 
  nstubs, eta region, GEM-CSC mathching included or not, etc...
- The macro uses tricky logic due to the fact that the samples (then ones with "gem") are splitted in 
  pT thresholds and need to be added to a single histogram (total rate)
- The results produced are for pat2 and pat8 (as the one general used is pat2 there is no need to call both 
  of them this can be implemented as a flag in the macro).
