# This script is an Imaris extension that performs 3D analysis of cell structures.
# It calculates various morphological features such as volume, surface area, thickness, sphericity, aspect ratio, convexity, contact area, and local curvature.
# The results are categorized into Low, Medium, and High bins based on percentiles.
# The user can choose which features to calculate and whether to save the results to an Excel file.

# Written by Jalal Al Rahbani, 2025
# GitHub: jalalrahbani

import ImarisLib
import numpy as np
import pandas as pd
import os
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist, squareform

def compute_curvature(vertices):
    """
    Approximate local curvature by computing the average distance 
    of each vertex to its nearest neighbors.
    """
    tree = KDTree(vertices)
    neighbors = tree.query(vertices, k=6)[1]  # Get 6 nearest neighbors
    curvatures = []

    for i, idx in enumerate(neighbors):
        local_points = vertices[idx]
        pairwise_distances = squareform(pdist(local_points))
        local_curvature = np.mean(pairwise_distances)
        curvatures.append(local_curvature)
    
    return np.array(curvatures)

def bin_data(values):
    """
    Categorize values into three bins: Low, Medium, High.
    """
    if len(values) < 3:
        return ["Medium"] * len(values)

    low_thresh = np.percentile(values, 33)
    high_thresh = np.percentile(values, 66)
    
    return [
        "Low" if val < low_thresh else "High" if val > high_thresh else "Medium"
        for val in values
    ]

def XT_MyImarisExtension(aImarisId):
    """
    Imaris extension with toggleable binning and curvature visualization.
    """
    # Connect to Imaris
    vImarisLib = ImarisLib.ImarisLib()
    vImaris = vImarisLib.GetApplication(aImarisId)
    if vImaris is None:
        print("Imaris connection failed!")
        return
    
    # Create a GUI dialog
    vFactory = vImaris.GetFactory()
    vDialog = vFactory.CreateDialog("Select Analysis Options")

    # Checkboxes for analysis selection
    vDialog.AddCheckbox("Calculate Volume", True)
    vDialog.AddCheckbox("Calculate Surface Area", True)
    vDialog.AddCheckbox("Calculate Thickness", True)
    vDialog.AddCheckbox("Calculate Sphericity", False)
    vDialog.AddCheckbox("Calculate Aspect Ratio", False)
    vDialog.AddCheckbox("Calculate Convexity", False)
    vDialog.AddCheckbox("Calculate Contact Area", False)
    vDialog.AddCheckbox("Calculate Local Curvature", True)
    vDialog.AddCheckbox("Enable Binning (Low, Medium, High)", False)  # New toggle
    vDialog.AddCheckbox("Save to Excel", True)

    vDialog.AddButton("OK", 1)
    vDialog.AddButton("Cancel", 0)

    # Show the dialog
    if not vDialog.Show():
        print("User canceled.")
        return
    
    # Retrieve user selections
    calculate_volume = vDialog.GetCheckbox("Calculate Volume")
    calculate_surface = vDialog.GetCheckbox("Calculate Surface Area")
    calculate_thickness = vDialog.GetCheckbox("Calculate Thickness")
    calculate_sphericity = vDialog.GetCheckbox("Calculate Sphericity")
    calculate_aspect_ratio = vDialog.GetCheckbox("Calculate Aspect Ratio")
    calculate_convexity = vDialog.GetCheckbox("Calculate Convexity")
    calculate_contact_area = vDialog.GetCheckbox("Calculate Contact Area")
    calculate_curvature = vDialog.GetCheckbox("Calculate Local Curvature")
    enable_binning = vDialog.GetCheckbox("Enable Binning (Low, Medium, High)")
    save_excel = vDialog.GetCheckbox("Save to Excel")

    vSurpassScene = vImaris.GetSurpassScene()
    if vSurpassScene is None:
        print("No Surpass scene found!")
        return

    all_results = []

    for i in range(vSurpassScene.GetNumberOfChildren()):
        vObject = vSurpassScene.GetChild(i)
        
        if vImaris.GetFactory().IsSurfaces(vObject):
            print(f"Processing: {vObject.GetName()}")
            vSurfaces = vImaris.GetFactory().ToSurfaces(vObject)
            vVertices = vSurfaces.GetVertices(0)

            if vVertices.shape[0] == 0:
                print(f"Skipping {vObject.GetName()} (no vertices)")
                continue

            result = {"Cell Name": vObject.GetName()}

            if calculate_volume:
                result["Volume"] = vSurfaces.GetVolume(0)

            if calculate_surface:
                result["Surface Area"] = vSurfaces.GetSurfaceArea(0)

            if calculate_thickness:
                result["Thickness"] = np.max(vVertices[:,2]) - np.min(vVertices[:,2])

            if calculate_sphericity:
                volume = vSurfaces.GetVolume(0)
                surface_area = vSurfaces.GetSurfaceArea(0)
                result["Sphericity"] = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area

            if calculate_aspect_ratio:
                eigvals = np.linalg.eigvals(np.cov(vVertices.T))
                result["Aspect Ratio"] = np.max(eigvals) / np.min(eigvals)

            if calculate_convexity:
                try:
                    hull = ConvexHull(vVertices)
                    convex_volume = hull.volume
                    result["Convexity"] = result["Volume"] / convex_volume
                except:
                    result["Convexity"] = np.nan

            if calculate_contact_area:
                min_z = np.min(vVertices[:,2])
                contact_points = vVertices[vVertices[:,2] == min_z]
                result["Contact Area"] = contact_points.shape[0]

            if calculate_curvature:
                curvature_values = compute_curvature(vVertices)
                curvature_min, curvature_max = np.min(curvature_values), np.max(curvature_values)
                normalized_curvature = (curvature_values - curvature_min) / (curvature_max - curvature_min)

                vRGBA = np.zeros((vVertices.shape[0], 4))
                vRGBA[:,0] = normalized_curvature  # Red (high curvature)
                vRGBA[:,2] = 1 - normalized_curvature  # Blue (low curvature)
                vRGBA[:,3] = 1  # Opacity
                vSurfaces.SetVertexColorsRGBA(0, vRGBA.flatten().tolist())

            all_results.append(result)

    df = pd.DataFrame(all_results)

    # Apply binning only if enabled
    if enable_binning:
        for col in ["Volume", "Surface Area", "Thickness", "Sphericity", "Aspect Ratio", "Convexity", "Contact Area"]:
            if col in df:
                df[f"{col} Category"] = bin_data(df[col])

    if save_excel:
        output_file = os.path.join(os.path.expanduser("~"), "Desktop", "Cell_Analysis.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Results saved to: {output_file}")

    print("3D Cell Analysis Completed!")
