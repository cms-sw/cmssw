#  ****************************************************************************
#  * Author:
#  *   Grzegorz Jędrzejowski,
#  ****************************************************************************

import ROOT
import os

# --- Configuration ---
# Root file
root_file_path = "simu_2018_protons.root"  
# Set the output directory for plots
output_directory = "./plots/plotsProtons/"
# Define the sub-folders to process
folders_to_process = ["rp3", "rp23", "rp103", "rp123"]
# Define the histograms to plot from each folder
histograms_to_plot = ["h_xi", "h2_th_y_vs_xi"]
# Define the additional plots to draw for multiRPPlots
multiRP_histograms_to_plot = ["arm0/h_xi", "arm1/h_xi"]

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# --- Main Plotting Logic ---
def plot_histograms_from_folders(file_path, base_dir, sub_dirs, hist_names, output_dir):
    """
    Accesses a ROOT file, reads histograms from specified sub-directories,
    and saves the plots to an output directory.
    """
    
    # Open the ROOT file
    root_file = ROOT.TFile.Open(file_path, "READ")
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open file {file_path}")
        return

    # Navigate to the base directory
    main_dir = root_file.Get(base_dir)
    if not main_dir:
        print(f"Error: Directory '{base_dir}' not found in file.")
        root_file.Close()
        return

    # Loop through each specified sub-folder
    for sub_dir_name in sub_dirs:
        print(f"Processing directory: {base_dir}/{sub_dir_name}")
        
        # Get the sub-directory
        sub_dir = main_dir.Get(sub_dir_name)
        if not sub_dir:
            print(f"Warning: Sub-directory '{sub_dir_name}' not found. Skipping.")
            continue

        # Loop through each histogram name
        for hist_name in hist_names:
            hist = sub_dir.Get(hist_name)
            hist.SetStats(0)
            hist.SetTitle(f"RP: {sub_dir_name} - {hist_name}")
            
            if not hist:
                print(f"Warning: Histogram '{hist_name}' not found in '{sub_dir_name}'. Skipping.")
                continue

            # Create a new canvas for each plot
            c = ROOT.TCanvas(f"c_{base_dir}_{sub_dir_name}_{hist_name}", f"{base_dir} - {sub_dir_name} - {hist_name}", 800, 600)

            # Draw the histogram based on its type
            if "h2_" in hist_name: # Check if it's a 2D histogram
                hist.Draw("COLZ")
            else: # Assume it's a 1D histogram
                hist.Draw()
            
            # Update the canvas to apply all drawing changes
            c.Update()

            # Save the plot
            safe_file_name = f"{base_dir}_{sub_dir_name}_{hist_name}".replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"{safe_file_name}.png")
            c.SaveAs(output_path)
            print(f"Plot saved to {output_path}")

            # Clean up the canvas object
            del c
    
    # Close the ROOT file
    root_file.Close()
    print("All plots generated and file closed.")

# --- Run the plotter for singleRPPlots ---
plot_histograms_from_folders(
    file_path=root_file_path,
    base_dir="singleRPPlots",
    sub_dirs=folders_to_process,
    hist_names=histograms_to_plot,
    output_dir=output_directory
)
# --- Run the plotter for multiRPPlots ---
plot_histograms_from_folders(
    file_path=root_file_path,
    base_dir="multiRPPlots",
    sub_dirs=["arm0", "arm1"],
    hist_names=["h_xi"],
    output_dir=output_directory
)