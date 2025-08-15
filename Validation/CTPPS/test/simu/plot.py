import ROOT
import os

# --- Configuration Section ---
file_paths = ['/afs/cern.ch/user/g/gjedrzej/private/mainTask/CMSSW_15_0_11/src/Validation/CTPPS/test/simu/simu_2018_Greg.root',
             '/afs/cern.ch/user/g/gjedrzej/private/mainTask/CMSSW_15_0_11/src/Validation/CTPPS/test/simu/simu_2018_64binsCalib.root',
             '/afs/cern.ch/user/g/gjedrzej/private/mainTask/CMSSW_15_0_11/src/Validation/CTPPS/test/simu/simu_2018_64binsPhys.root']

file_labels = ['Unfiltered', 'Calibration', 'Physics']

#histogram names
histogram_names = ['Example Histogram', 'Theta Degrees', 'Phi', 'Energy', 'Pt', 'Xi']

#setup
output_directory = './plots/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

#Plotting 
for hist_name in histogram_names:
    print(f"Processing histogram: {hist_name}")

    c1 = ROOT.TCanvas('c1', hist_name, 1200, 400)
    c1.Divide(3, 1)
    
    files = []
    hists = []

    for i, path in enumerate(file_paths):
        try:
            f = ROOT.TFile.Open(path)
            if not f or f.IsZombie():
                print(f"Error: Could not open file {path}")
                continue
            files.append(f)

            hist = f.Get(hist_name)
            if not hist:
                print(f"Error: Histogram '{hist_name}' not found in {path}")
                continue
            hists.append(hist)

            c1.cd(i + 1)
            hist.SetTitle(file_labels[i])
            hist.Draw()

        except Exception as e:
            print(f"An error occurred: {e}")

    c1.Update()
    
    # Sanitize the histogram name for the filename
    safe_hist_name = hist_name.replace(" ", "_").replace(";", "")
    output_filename = os.path.join(output_directory, f"{safe_hist_name}_comparison.png")
    c1.SaveAs(output_filename)
    
    print(f"Comparison plot saved to {output_filename}")
    
    # Clean up the canvas and close files for the next loop iteration
    del c1
    for f in files:
        f.Close()

print("All histograms have been processed.")