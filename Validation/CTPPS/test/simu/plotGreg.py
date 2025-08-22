#  ****************************************************************************
#  * Author:
#  *   Grzegorz Jędrzejowski,
#  ****************************************************************************


import ROOT
import os

# --- Configuration Section ---
file_paths = ['simu_2018_Greg.root',
              'simu_2018_64binsCalib.root',
              'simu_2018_64binsPhys.root']
              
file_labels = ['Unfiltered', 'Calibration', 'Physics']

#histogram names
histogram_names = ['Example Histogram', 'Theta', 'Phi', 'Energy', 'Pt', 'Xi']
histogram_3d_names = ['Xi_Pt_Phi', 'Xi_Theta_Phi']


# Phi slices for projections (e.g., in radians)
phi_slices = [(-5.0, -2.5), (-2.5, 0.0), (0.0, 2.5), (2.5, 5.0)]
phi_slice_labels = ['Phi_-5.0_to_-2.5', 'Phi_-2.5_to_0.0', 'Phi_0.0_to_2.5', 'Phi_2.5_to_5.0']

#setup
output_directory = './plots/plotsGreg/'
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



for hist_3d_name in histogram_3d_names:
    print(f"Processing 3D histogram: {hist_3d_name}")

    # 2D projection for all phi
    c_all_phi = ROOT.TCanvas('c_all_phi', f'{hist_3d_name} All Phi', 1200, 400)
    c_all_phi.Divide(3, 1)

    # 2D projections for phi slices
    c_phi_slices = ROOT.TCanvas('c_phi_slices', f'{hist_3d_name} Phi Slices', 3600, 1200 * len(phi_slices))
    c_phi_slices.Divide(3, len(phi_slices))

    files = []
    
    # Create lists to hold the projected histograms to prevent them from being garbage collected
    all_phi_projections = []
    slice_projections = []

    for i, path in enumerate(file_paths):
        try:
            f = ROOT.TFile.Open(path)
            if not f or f.IsZombie():
                print(f"Error: Could not open file {path}")
                continue
            files.append(f)
            
            hist_3d = f.Get(hist_3d_name)
            if not hist_3d:
                print(f"Error: 3D histogram '{hist_3d_name}' not found in {path}")
                continue
            
            # --- Plotting All Phi Projection ---
            c_all_phi.cd(i + 1)
            hist_3d.GetZaxis().SetRange(1, hist_3d.GetNbinsZ())
            
            # Create the projected histogram and store it in a list
            # proj_all_phi = hist_3d.Project3D("xy")
            proj_all_phi = hist_3d.Project3D("yx").Clone(f"{hist_3d_name}_allphi_{i}")
            all_phi_projections.append(proj_all_phi)
            
            # Set title and axis labels on the newly created object
            proj_all_phi.SetTitle(f'{file_labels[i]} (All Phi)')
            proj_all_phi.GetXaxis().SetTitle("Xi")
            if "Pt" in hist_3d_name:
                proj_all_phi.GetYaxis().SetTitle("Pt")
            elif "Theta" in hist_3d_name:
                proj_all_phi.GetYaxis().SetTitle("Theta")
            
            proj_all_phi.Draw("COLZ")
            
            # --- Plotting Phi Slices ---
            for j, (phi_min, phi_max) in enumerate(phi_slices):
                c_phi_slices.cd(i + 1 + 3 * j)
                
                phi_axis = hist_3d.GetZaxis()
                bin_min = phi_axis.FindBin(phi_min)
                bin_max = phi_axis.FindBin(phi_max)
                
                hist_3d.GetZaxis().SetRange(bin_min, bin_max)
                
                # Create the projected slice and store it in a list
                # proj_slice = hist_3d.Project3D("xy")
                proj_slice = hist_3d.Project3D("yx").Clone(f"{hist_3d_name}_phi_{i}_{j}")
                slice_projections.append(proj_slice)
                
                hist_3d.GetZaxis().SetRange(1, hist_3d.GetNbinsZ())

                proj_slice.SetTitle(f'{file_labels[i]} ({phi_slice_labels[j]})')
                proj_slice.GetXaxis().SetTitle("Xi")
                proj_slice.SetStats(0)
                if "Pt" in hist_3d_name:
                    proj_slice.GetYaxis().SetTitle("Pt")
                elif "Theta" in hist_3d_name:
                    proj_slice.GetYaxis().SetTitle("Theta")
                
                proj_slice.Draw("COLZ")
                c_phi_slices.Modified()
                c_phi_slices.Update()
                
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save the plots
    c_all_phi.Update()
    c_all_phi.Modified()
    safe_hist_name_3d = hist_3d_name.replace(" ", "_")  
    output_filename_all_phi = os.path.join(output_directory, f"{safe_hist_name_3d}_all_phi_comparison.png")
    c_all_phi.SaveAs(output_filename_all_phi)
    print(f"All Phi comparison plot saved to {output_filename_all_phi}")

    c_phi_slices.Update()
    c_phi_slices.Modified()
    output_filename_slices = os.path.join(output_directory, f"{safe_hist_name_3d}_phi_slices_comparison.png")
    c_phi_slices.SaveAs(output_filename_slices)
    print(f"Phi slices comparison plot saved to {output_filename_slices}")

    # Clean up
    del c_all_phi
    del c_phi_slices
    for f in files:
        f.Close()




# --- Plotting 1D Histograms on the same canvas ---
for hist_name in histogram_names:
    print(f"Processing 1D histogram: {hist_name}")

    c1 = ROOT.TCanvas('c1', hist_name, 800, 600)
    
    files = []
    hists_to_keep = [] # List to prevent histograms from being garbage collected
    
    # Define colors and line styles
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2]
    line_styles = [1, 2, 3] # Solid, dashed, dotted
    
    # Create the legend
    legend = ROOT.TLegend(0.8, 0.8, 1.0, 1.0)

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


            # Set styling
            hist.SetStats(0)
            hist.SetLineColor(colors[i])
            hist.SetLineStyle(line_styles[i])
            hist.SetLineWidth(2)
            hist.SetTitle(f'Comparison of {hist_name}')
            
            hists_to_keep.append(hist)
            legend.AddEntry(hist, file_labels[i], "l")

            # Draw the first histogram normally, subsequent ones with "SAME"
            if i == 0:
                hist.Draw()
                hist.GetXaxis().SetTitle(hist_name)
                hist.GetYaxis().SetTitle("Entries")
            else:
                hist.Draw("SAME")

        except Exception as e:
            print(f"An error occurred: {e}")

    # Draw the legend and update the canvas
    legend.Draw()
    c1.Update()
    
    # Save the plot
    safe_hist_name = hist_name.replace(" ", "_").replace(";", "")
    output_filename = os.path.join(output_directory, f"{safe_hist_name}_overlay_comparison.png")
    c1.SaveAs(output_filename)
    
    print(f"Comparison plot saved to {output_filename}")
    
    # Clean up
    del c1
    for f in files:
        f.Close()










print("All histograms have been processed.")
