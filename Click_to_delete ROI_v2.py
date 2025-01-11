# -----------------------------------------------------------------------------
# Script Name: Click-to-Delete ROI and Bulk Delete within Selection
# Description: 
#     This script enhances ROI (Region of Interest) management in Fiji/ImageJ by
#     providing two main functionalities:
#         1. **Click-to-Delete:** Allows users to delete individual ROIs by
#            left-clicking directly on them.
#         2. **Bulk Delete Within Selection:** Enables users to delete multiple ROIs
#            by drawing a rectangular selection around the desired area encompassing
#            the ROIs to be removed.
#
# Features:
#     - Activate and deactivate the click-to-delete listener using GUI buttons.
#     - Confirmation dialogs to prevent accidental deletions.
#     - Logging of actions for transparency and debugging purposes.
#
# Requirements:
#     - Fiji/ImageJ with Python (Jython) scripting support.
#     - ROIs added to the ROI Manager.
#
# Usage Instructions:
#     1. **Load Image and ROIs:**
#         - Open your image in Fiji/ImageJ.
#         - Draw ROIs using any ROI tool (e.g., Rectangle, Oval).
#         - Add ROIs to the ROI Manager via `Analyze > Tools > ROI Manager` or by clicking the "Add" button.
#
#     2. **Run the Script:**
#         - Navigate to `Plugins > Scripting > New > Python` in Fiji/ImageJ.
#         - Paste this script into the Script Editor.
#         - Click the "Run" button (green ▶️ icon).
#
#     3. **Using the GUI:**
#         - A small window titled "RoiClickListener Control" will appear with three buttons:
#             - **Activate Listener:** Enables the click-to-delete functionality.
#             - **Deactivate Listener:** Disables the click-to-delete functionality.
#             - **Delete ROIs Within Selection:** Deletes all ROIs entirely within a drawn rectangle.
#
#     4. **Deleting Individual ROIs:**
#         - Click the "Activate Listener" button.
#         - Left-click on any ROI in the image to prompt deletion.
#
#     5. **Deleting Multiple ROIs:**
#         - Select the Rectangle Selection Tool from the toolbar.
#         - Draw a rectangle around the area containing the ROIs you wish to delete.
#         - Click the "Delete ROIs Within Selection" button in the GUI.
#
#     6. **Stopping the Listener:**
#         - Click the "Deactivate Listener" button to stop the click-to-delete functionality.
#
# Author: Jalal Al Rahbani
# Date: January 9, 2025
# Version: 1.0
# -----------------------------------------------------------------------------

from ij import IJ
from ij.plugin.frame import RoiManager
from java.awt.event import MouseAdapter, MouseEvent
from javax.swing import JButton, JFrame, JPanel, SwingUtilities
from ij.gui import Roi

# Initialize ROI Manager
rm = RoiManager.getInstance()
if rm is None:
    rm = RoiManager()

# Get the current image and its window
imp = IJ.getImage()
image_window = imp.getWindow()
canvas = image_window.getCanvas()

# Define the Mouse Listener for Click-to-Delete
class RoiClickListener(MouseAdapter):
    def mouseClicked(self, event):
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

# Function to delete ROIs within a rectangle selection
def delete_rois_within_selection():
    # Get the current selection
    selection = imp.getRoi()
    if selection is None:
        IJ.showMessage("No Selection", "Please draw a rectangle selection on the image.")
        return
    if not isinstance(selection, Roi):
        IJ.showMessage("Invalid Selection", "Please use the Rectangle Selection tool.")
        return

    # Ensure the selection is a rectangle
    if selection.getType() != Roi.RECTANGLE:
        IJ.showMessage("Invalid Selection", "Please use the Rectangle Selection tool.")
        return

    # Get the bounds of the selection
    sel_bounds = selection.getBounds()
    sel_x = sel_bounds.x
    sel_y = sel_bounds.y
    sel_w = sel_bounds.width
    sel_h = sel_bounds.height

    IJ.log("Selection bounds: x={}, y={}, width={}, height={}".format(sel_x, sel_y, sel_w, sel_h))

    # Define the selection rectangle as a Rectangle object
    selection_rect = sel_bounds  # java.awt.Rectangle

    # Retrieve all ROIs from the ROI Manager
    rois = rm.getRoisAsArray()

    # List to hold indices of ROIs to delete
    rois_to_delete = []

    # Iterate over ROIs to check if they are within the selection
    for idx, roi in enumerate(rois):
        roi_bounds = roi.getBounds()
        if selection_rect.contains(roi_bounds):
            rois_to_delete.append(idx)
            IJ.log("ROI {} is within the selection and marked for deletion.".format(idx))
        # To delete ROIs that intersect with the selection, use the following:
        # elif selection_rect.intersects(roi_bounds):
        #     rois_to_delete.append(idx)
        #     IJ.log("ROI {} intersects with the selection and is marked for deletion.".format(idx))

    if not rois_to_delete:
        IJ.showMessage("No ROIs Found", "No ROIs are within the selected area.")
        return

    # Confirm deletion with the user
    num_rois = len(rois_to_delete)
    confirm = IJ.showMessageWithCancel("Confirm Deletion",
                                       "Are you sure you want to delete {} ROI(s)?".format(num_rois))
    if not confirm:
        IJ.log("Deletion canceled by the user.")
        return

    # Delete ROIs starting from the highest index to prevent reindexing issues
    rois_to_delete.sort(reverse=True)
    for idx in rois_to_delete:
        rm.select(idx)
        rm.runCommand("Delete")
        IJ.log("Deleted ROI at index: {}".format(idx))

    # Refresh the image
    imp.updateAndDraw()

    IJ.showMessage("Deletion Complete", "Deleted {} ROI(s) within the selected area.".format(num_rois))

# Create a simple GUI with Activate, Deactivate, and Delete Within Selection buttons
def create_gui():
    def on_activate(event):
        add_listener()
    
    def on_deactivate(event):
        remove_listener()
    
    def on_delete_within(event):
        delete_rois_within_selection()
    
    # Create a JFrame
    frame = JFrame("RoiClickListener Control")
    frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE)
    
    # Create a panel and add buttons
    panel = JPanel()
    activate_button = JButton("Activate Listener")
    deactivate_button = JButton("Deactivate Listener")
    delete_within_button = JButton("Delete ROIs Within Selection")
    
    activate_button.addActionListener(on_activate)
    deactivate_button.addActionListener(on_deactivate)
    delete_within_button.addActionListener(on_delete_within)
    
    panel.add(activate_button)
    panel.add(deactivate_button)
    panel.add(delete_within_button)
    
    frame.getContentPane().add(panel)
    frame.pack()
    frame.setVisible(True)

# Run the GUI on the Event Dispatch Thread
SwingUtilities.invokeLater(create_gui)

# Inform the user
IJ.log("RoiClickListener Control GUI opened.")
IJ.log("1. Click 'Activate Listener' to enable click-to-delete functionality.")
IJ.log("2. Left-click on any ROI to delete it.")
IJ.log("3. To delete multiple ROIs within a selected area,")
IJ.log("   a. Select the Rectangle Selection Tool and draw a rectangle around desired ROIs.")
IJ.log("   b. Click 'Delete ROIs Within Selection' to remove them.")
IJ.log("4. Click 'Deactivate Listener' to disable click-to-delete functionality.")
