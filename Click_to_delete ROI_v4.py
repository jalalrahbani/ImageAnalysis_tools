# -----------------------------------------------------------------------------
# Script Name: Enhanced ROI Management Tool for Fiji/ImageJ
# Description: 
#     This script enhances ROI (Region of Interest) management in Fiji/ImageJ by
#     providing the following functionalities:
#         1. **Click-to-Delete:** Allows users to delete individual ROIs by
#            left-clicking directly on them.
#         2. **Bulk Delete Within Any Selection:** Enables users to delete multiple ROIs
#            by drawing any type of selection (Rectangle, Oval/Circle, Freeform) around
#            the desired area encompassing the ROIs to be removed.
#         3. **Selection Tool Switching:** Provides GUI buttons to switch between
#            different selection tools (Rectangle, Oval, Freeform).
#         4. **"Frenzy" Mode:** Adds a mode that bypasses deletion confirmation dialogs
#            with a warning alert to prevent accidental deletions.
#         5. **Smooth ROIs:** Allows users to smooth all remaining ROIs for a refined appearance.
#
# Features:
#     - Activate and deactivate the click-to-delete listener using GUI buttons.
#     - Select the type of selection tool (Rectangle, Oval, Freeform) via GUI buttons.
#     - Toggle "Frenzy" mode to delete ROIs without confirmation.
#     - Smooth all remaining ROIs with a single button click.
#     - Confirmation dialogs to prevent accidental deletions (unless in Frenzy mode).
#     - Logging of actions for transparency and debugging purposes.
#
# Requirements:
#     - Fiji/ImageJ with Python (Jython) scripting support.
#     - ROIs added to the ROI Manager.
#
# Usage Instructions:
#     1. **Load Image and ROIs:**
#         - Open your image in Fiji/ImageJ.
#         - Draw ROIs using any ROI tool (e.g., Rectangle, Oval, Freehand).
#         - Add ROIs to the ROI Manager via `Analyze > Tools > ROI Manager` or by clicking the "Add" button.
#
#     2. **Run the Script:**
#         - Navigate to `Plugins > Scripting > New > Python` in Fiji/ImageJ.
#         - Paste this script into the Script Editor.
#         - Click the "Run" button (green ▶️ icon).
#
#     3. **Using the GUI:**
#         - A window titled **"Enhanced ROI Manager Control"** will appear with the following buttons:
#             - **Activate Listener:** Enables the click-to-delete functionality.
#             - **Deactivate Listener:** Disables the click-to-delete functionality.
#             - **Select Rectangle Tool:** Switches to the Rectangle selection tool.
#             - **Select Oval Tool:** Switches to the Oval/Circle selection tool.
#             - **Select Freeform Tool:** Switches to the Freeform (Freehand) selection tool.
#             - **Frenzy Mode:** Toggles the Frenzy mode for automatic deletions without confirmation.
#             - **Delete ROIs Within Selection:** Deletes all ROIs within the current selection.
#             - **Smooth ROIs:** Smooths all remaining ROIs for a refined appearance.
#
#     4. **Deleting Individual ROIs:**
#         - Click the **"Activate Listener"** button.
#         - Left-click on any ROI in the image to prompt deletion (confirmation dialog appears unless Frenzy mode is active).
#
#     5. **Deleting Multiple ROIs Within a Selection:**
#         - **a.** Select the desired selection tool using the corresponding GUI button (Rectangle, Oval, Freeform).
#         - **b.** Draw the selection around the ROIs you wish to delete.
#         - **c.** Click the **"Delete ROIs Within Selection"** button in the GUI.
#             - **Note:** If "Frenzy" mode is active, deletions will occur without confirmation.
#
#     6. **Activating Frenzy Mode:**
#         - Click the **"Frenzy Mode"** button.
#         - A warning dialog will appear. Click **"OK"** to activate Frenzy mode.
#         - In Frenzy mode, ROIs will be deleted immediately upon clicking without confirmation.
#         - A warning message will be logged in the Log window.
#
#     7. **Deactivating Frenzy Mode:**
#         - Click the **"Frenzy Mode"** button again.
#         - A confirmation dialog will appear. Click **"OK"** to deactivate Frenzy mode.
#         - A confirmation message will be logged in the Log window.
#
#     8. **Smoothing ROIs:**
#         - Click the **"Smooth ROIs"** button.
#         - All remaining ROIs in the ROI Manager will be smoothed for a refined appearance.
#         - A confirmation message will be displayed upon completion.
#
#     9. **Deactivating the Listener:**
#         - Click the **"Deactivate Listener"** button to stop the click-to-delete functionality.
#
#    10. **Closing the GUI:**
#         - To close the **"Enhanced ROI Manager Control"** GUI window, simply close the window by clicking the "X" button.
#
# Author: Jalal Al Rahbani
# Date: 01/16/2025
# Version: 4
# -----------------------------------------------------------------------------

from ij import IJ
from ij.plugin.frame import RoiManager
from java.awt.event import MouseAdapter, MouseEvent
from javax.swing import JButton, JFrame, JPanel, SwingUtilities, JOptionPane, BoxLayout
from ij.gui import Roi, PolygonRoi
from java.awt import Color, Polygon

# Initialize ROI Manager
rm = RoiManager.getInstance()
if rm is None:
    rm = RoiManager()

# Get the current image and its window
imp = IJ.getImage()
image_window = imp.getWindow()
canvas = image_window.getCanvas()

# Global variable to track Frenzy mode
frenzy_mode = False

# Define the Mouse Listener for Click-to-Delete
class RoiClickListener(MouseAdapter):
    def mouseClicked(self, event):
        global frenzy_mode
        # Only respond to left-clicks
        if event.getButton() != MouseEvent.BUTTON1:
            return
        
        # Get the click coordinates in image space
        x_click = canvas.offScreenX(event.getX())
        y_click = canvas.offScreenY(event.getY())
        
        # Retrieve all ROIs from the ROI Manager
        rois = rm.getRoisAsArray()
        
        # Iterate over ROIs to check if the click is inside any
        for idx, roi in enumerate(rois):
            if roi.contains(x_click, y_click):
                # Select the ROI in the ROI Manager
                rm.select(idx)
                
                if frenzy_mode:
                    # Delete without confirmation
                    rm.runCommand("Delete")
                    IJ.log("Frenzy Mode: Deleted ROI at index: {}".format(idx))
                else:
                    # Confirm deletion using compatible string formatting
                    confirm = IJ.showMessageWithCancel("Delete ROI",
                                                       "Delete ROI {}?".format(idx + 1))
                    if confirm:
                        rm.runCommand("Delete")
                        IJ.log("Deleted ROI at index: {}".format(idx))
                # Redraw the image to update ROI display
                imp.updateAndDraw()
                return  # Exit after deleting the first matching ROI
        
        # If no ROI was clicked
        IJ.log("Clicked outside of any ROI.")

# Instantiate the Mouse Listener
listener = RoiClickListener()

# Function to add the click listener
def add_listener():
    canvas.addMouseListener(listener)
    IJ.log("RoiClickListener activated. Left-click on an ROI to delete it.")

# Function to remove the click listener
def remove_listener():
    canvas.removeMouseListener(listener)
    IJ.log("RoiClickListener deactivated.")

# Function to switch selection tools
def set_selection_tool(tool_name):
    # Possible tool names: "rectangle", "oval", "freehand"
    valid_tools = ["rectangle", "oval", "freehand"]
    if tool_name.lower() not in valid_tools:
        IJ.showMessage("Invalid Tool", "Selected tool is not supported.")
        return
    IJ.setTool(tool_name.lower())
    IJ.log("Selection tool set to: {}".format(tool_name.capitalize()))

# Function to delete ROIs within the current selection (any selection type)
def delete_rois_within_selection():
    # Get the current selection
    selection = imp.getRoi()
    if selection is None:
        IJ.showMessage("No Selection", "Please draw a selection on the image.")
        return

    # Get the bounds of the selection
    sel_bounds = selection.getBounds()
    sel_x = sel_bounds.x
    sel_y = sel_bounds.y
    sel_w = sel_bounds.width
    sel_h = sel_bounds.height

    IJ.log("Selection bounds: x={}, y={}, width={}, height={}".format(sel_x, sel_y, sel_w, sel_h))

    # Define the selection ROI
    selection_roi = selection

    # Retrieve all ROIs from the ROI Manager
    rois = rm.getRoisAsArray()

    # List to hold indices of ROIs to delete
    rois_to_delete = []

    # Iterate over ROIs to check if they are within the selection
    for idx, roi in enumerate(rois):
        roi_bounds = roi.getBounds()
        # Check if ROI is entirely within the selection
        if (selection_roi.contains(roi_bounds.x, roi_bounds.y) and
            selection_roi.contains(roi_bounds.x + roi_bounds.width, roi_bounds.y + roi_bounds.height)):
            rois_to_delete.append(idx)
            IJ.log("ROI {} is entirely within the selection and marked for deletion.".format(idx))
        # Alternatively, to delete ROIs that intersect with the selection:
        # elif selection_roi.intersects(roi_bounds.x, roi_bounds.y, roi_bounds.width, roi_bounds.height):
        #     rois_to_delete.append(idx)
        #     IJ.log("ROI {} intersects with the selection and is marked for deletion.".format(idx))

    if not rois_to_delete:
        IJ.showMessage("No ROIs Found", "No ROIs are within the selected area.")
        return

    # Confirmation dialog (unless Frenzy mode is active)
    global frenzy_mode
    if frenzy_mode:
        confirm = True
    else:
        num_rois = len(rois_to_delete)
        confirm = IJ.showMessageWithCancel("Confirm Deletion",
                                           "Are you sure you want to delete {} ROI(s)?".format(num_rois))
    
    if not confirm:
        if not frenzy_mode:
            IJ.log("Deletion canceled by the user.")
        return

    # Delete ROIs starting from the highest index to prevent reindexing issues
    rois_to_delete.sort(reverse=True)
    for idx in rois_to_delete:
        rm.select(idx)
        rm.runCommand("Delete")
        if frenzy_mode:
            IJ.log("Frenzy Mode: Deleted ROI at index: {}".format(idx))
        else:
            IJ.log("Deleted ROI at index: {}".format(idx))

    # Refresh the image
    imp.updateAndDraw()

    if frenzy_mode:
        IJ.log("Frenzy Mode: Deleted {} ROI(s) within the selected area.".format(len(rois_to_delete)))
    else:
        IJ.showMessage("Deletion Complete", "Deleted {} ROI(s) within the selected area.".format(len(rois_to_delete)))

# Function to toggle Frenzy mode
def toggle_frenzy():
    global frenzy_mode
    if not frenzy_mode:
        # Activate Frenzy mode
        frenzy_mode = True
        # Show warning message
        JOptionPane.showMessageDialog(None, 
                                      "Frenzy Mode Activated!\nAll ROI deletions will occur without confirmation.",
                                      "Frenzy Mode Warning",
                                      JOptionPane.WARNING_MESSAGE)
        IJ.log("Frenzy Mode Activated: Deletions will occur without confirmation.")
    else:
        # Deactivate Frenzy mode
        frenzy_mode = False
        # Inform the user
        JOptionPane.showMessageDialog(None, 
                                      "Frenzy Mode Deactivated!\nROI deletions will now require confirmation.",
                                      "Frenzy Mode Warning",
                                      JOptionPane.INFORMATION_MESSAGE)
        IJ.log("Frenzy Mode Deactivated: Deletions will now require confirmation.")

# Function to smooth all ROIs
def smooth_rois():
    rois = rm.getRoisAsArray()
    if not rois:
        IJ.showMessage("No ROIs", "There are no ROIs to smooth.")
        return

    for idx, roi in enumerate(rois):
        # Only smooth polygonal ROIs (e.g., Freehand, Polygon)
        if isinstance(roi, Roi) and (roi.getType() == Roi.POLYGON or roi.getType() == Roi.FREEROI):
            poly = roi.getPolygon()
            if poly.npoints < 3:
                IJ.log("ROI at index {} is too simple to smooth and was skipped.".format(idx))
                continue
            # Apply smoothing by creating a new smoothed polygon
            smoothed_poly = smooth_polygon(poly, iterations=2)
            # Create a new PolygonRoi from the smoothed polygon
            new_roi = PolygonRoi(smoothed_poly.xpoints, smoothed_poly.ypoints, smoothed_poly.npoints, Roi.POLYGON)
            # Replace the old ROI with the smoothed one
            rm.addRoi(new_roi)
            rm.select(idx)
            rm.runCommand("Delete")
            IJ.log("Smoothed ROI at index: {}".format(idx))
        else:
            IJ.log("ROI at index {} is not a polygonal ROI and was skipped.".format(idx))
    
    # Refresh the image
    imp.updateAndDraw()
    IJ.showMessage("Smoothing Complete", "All polygonal ROIs have been smoothed.")
    IJ.log("All polygonal ROIs have been smoothed.")

# Simple smoothing function for polygons
def smooth_polygon(polygon, iterations=2):
    smoothed = Polygon()
    n = polygon.npoints
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        new_x = (polygon.xpoints[prev_idx] + polygon.xpoints[i] + polygon.xpoints[next_idx]) / 3
        new_y = (polygon.ypoints[prev_idx] + polygon.ypoints[i] + polygon.ypoints[next_idx]) / 3
        smoothed.addPoint(int(new_x), int(new_y))
    if iterations > 1:
        return smooth_polygon(smoothed, iterations - 1)
    return smoothed

# Create a simple GUI with multiple rows of buttons
def create_gui():
    # Create a JFrame
    frame = JFrame("Enhanced ROI Manager Control")
    frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE)
    
    # Create a main panel with BoxLayout (Y_AXIS)
    main_panel = JPanel()
    main_panel.setLayout(BoxLayout(main_panel, BoxLayout.Y_AXIS))
    
    # First row: Activate and Deactivate Listener
    panel1 = JPanel()
    activate_button = JButton("Activate Listener")
    deactivate_button = JButton("Deactivate Listener")
    panel1.add(activate_button)
    panel1.add(deactivate_button)
    main_panel.add(panel1)
    
    # Second row: Selection Tool Buttons
    panel2 = JPanel()
    rectangle_button = JButton("Select Rectangle Tool")
    oval_button = JButton("Select Oval Tool")
    freehand_button = JButton("Select Freeform Tool")
    panel2.add(rectangle_button)
    panel2.add(oval_button)
    panel2.add(freehand_button)
    main_panel.add(panel2)
    
    # Third row: Frenzy Mode and Delete Within Selection
    panel3 = JPanel()
    frenzy_button = JButton("Frenzy Mode")
    delete_within_button = JButton("Delete ROIs Within Selection")
    panel3.add(frenzy_button)
    panel3.add(delete_within_button)
    main_panel.add(panel3)
    
    # Fourth row: Smooth ROIs
    panel4 = JPanel()
    smooth_button = JButton("Smooth ROIs")
    panel4.add(smooth_button)
    main_panel.add(panel4)
    
    # Add all panels to the main frame
    frame.getContentPane().add(main_panel)
    frame.pack()
    frame.setVisible(True)
    
    # Define button actions
    activate_button.addActionListener(lambda event: add_listener())
    deactivate_button.addActionListener(lambda event: remove_listener())
    rectangle_button.addActionListener(lambda event: set_selection_tool("rectangle"))
    oval_button.addActionListener(lambda event: set_selection_tool("oval"))
    freehand_button.addActionListener(lambda event: set_selection_tool("freehand"))
    frenzy_button.addActionListener(lambda event: toggle_frenzy())
    delete_within_button.addActionListener(lambda event: delete_rois_within_selection())
    smooth_button.addActionListener(lambda event: smooth_rois())

# Run the GUI on the Event Dispatch Thread
SwingUtilities.invokeLater(create_gui)

# Inform the user
IJ.log("Enhanced ROI Manager Control GUI opened.")
IJ.log("1. Click 'Activate Listener' to enable click-to-delete functionality.")
IJ.log("2. Use selection tool buttons to choose your selection type (Rectangle, Oval, Freeform).")
IJ.log("3. Click 'Frenzy Mode' to toggle automatic deletions without confirmation.")
IJ.log("4. Left-click on any ROI to delete it (confirmation required unless Frenzy mode is active).")
IJ.log("5. To delete multiple ROIs within a selection,")
IJ.log("   a. Select the desired selection tool and draw the selection around ROIs.")
IJ.log("   b. Click 'Delete ROIs Within Selection' to remove them.")
IJ.log("6. Click 'Smooth ROIs' to smooth all remaining polygonal ROIs.")
IJ.log("7. Click 'Deactivate Listener' to disable click-to-delete functionality.")
