import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify, vtk as vtk_widgets

# -----------------------------------------------------------------------------
# VTK Pipeline
# -----------------------------------------------------------------------------
source = vtk.vtkRTAnalyticSource()
source.SetWholeExtent(-10, 10, -10, 10, -10, 10)
source.Update()

dataset = vtk.vtkImageData()
dataset.DeepCopy(source.GetOutput())

scalars = dataset.GetPointData().GetScalars()
arr = vtk_to_numpy(scalars).astype(np.float32)

original_arr = vtk_to_numpy(scalars).astype(np.float32).copy()
original_min = original_arr.min()
original_max = original_arr.max()

# ----------------- Pipeline 1: vtkLookupTable -----------------
mapper1 = vtk.vtkDataSetMapper()
mapper1.SetInputData(dataset)
mapper1.SetUseLookupTableScalarRange(True)

actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)

lut = vtk.vtkLookupTable()
mapper1.SetLookupTable(lut)

scalar_bar1 = vtk.vtkScalarBarActor()
scalar_bar1.SetLookupTable(lut)
scalar_bar1.SetTitle("vtkLookupTable")
scalar_bar1.SetNumberOfLabels(5)
scalar_bar1.SetDrawBelowRangeSwatch(True)
scalar_bar1.SetDrawAboveRangeSwatch(True)
scalar_bar1.SetDrawNanAnnotation(True)
scalar_bar1.SetDrawAnnotations(True)
scalar_bar1.SetDrawTickLabels(False)

renderer1 = vtk.vtkRenderer()
renderer1.AddActor(actor1)
renderer1.AddViewProp(scalar_bar1)
renderer1.ResetCamera()

renderWindow1 = vtk.vtkRenderWindow()
renderWindow1.AddRenderer(renderer1)

renderWindowInteractor1 = vtk.vtkRenderWindowInteractor()
renderWindowInteractor1.SetRenderWindow(renderWindow1)

# ----------------- Pipeline 2: vtkColorTransferFunction -----------------
mapper2 = vtk.vtkDataSetMapper()
mapper2.SetInputData(dataset)
mapper2.SetUseLookupTableScalarRange(True)

actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)

ctf = vtk.vtkColorTransferFunction()
mapper2.SetLookupTable(ctf)

scalar_bar2 = vtk.vtkScalarBarActor()
scalar_bar2.SetLookupTable(ctf)
scalar_bar2.SetTitle("vtkColorTransferFunction")
scalar_bar2.SetNumberOfLabels(5)
scalar_bar2.SetDrawBelowRangeSwatch(True)
scalar_bar2.SetDrawAboveRangeSwatch(True)
scalar_bar2.SetDrawNanAnnotation(True)
scalar_bar2.SetDrawAnnotations(True)
scalar_bar2.SetDrawTickLabels(False)

renderer2 = vtk.vtkRenderer()
renderer2.AddActor(actor2)
renderer2.AddViewProp(scalar_bar2)
renderer2.ResetCamera()

renderWindow2 = vtk.vtkRenderWindow()
renderWindow2.AddRenderer(renderer2)

renderWindowInteractor2 = vtk.vtkRenderWindowInteractor()
renderWindowInteractor2.SetRenderWindow(renderWindow2)

# -----------------------------------------------------------------------------
# Trame Server Setup
# -----------------------------------------------------------------------------
server = get_server()
state, ctrl = server.state, server.controller

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) >= 6:
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (1.0, 1.0, 1.0)

def hex_to_rgba(hex_color):
    rgb = hex_to_rgb(hex_color)
    return rgb + (1.0,) # Alpha = 1

@state.change("data_range")
def update_data_array(data_range, **kwargs):
    d_min, d_max = data_range
    # Scale array to [d_min, d_max]
    arr = (original_arr - original_min) / (original_max - original_min) * (d_max - d_min) + d_min
    
    # Reshape and apply out of bounds/NaN chunks
    arr_3d = arr.reshape((21, 21, 21))
    arr_3d[10:21, 10:21, 10:21] = np.nan 
    arr_3d[10:21, 0:10, 10:21] = d_min - 5.0
    arr_3d[0:10, 10:21, 10:21] = d_max + 5.0
    
    # Push back to dataset
    new_scalars = numpy_to_vtk(arr)
    new_scalars.SetName(scalars.GetName())
    dataset.GetPointData().SetScalars(new_scalars)
    
    dataset.GetPointData().Modified()
    dataset.Modified()
    mapper1.Modified()
    mapper2.Modified()
    
    renderWindow1.Render()
    renderWindow2.Render()
    if hasattr(ctrl, "view_update_1"):
        ctrl.view_update_1()
    if hasattr(ctrl, "view_update_2"):
        ctrl.view_update_2()

@state.change("use_above", "above_color", "use_below", "below_color", "nan_color", "use_log", "color_range", "show_data")
def update_colormap(use_above, above_color, use_below, below_color, nan_color, use_log, color_range, show_data, **kwargs):
    min_range, max_range = color_range
    
    # Implement ParaView's AdjustRangeForLog logic
    if use_log and (min_range <= 0 or max_range <= 0):
        if max_range <= 0:
            min_range = 1.0e-4
            max_range = 1.0
        else:
            min_range = max_range * 0.0001
            min_range = min_range if min_range < 1.0 else 1.0
            
        # Push the fixed range back to the UI state.
        # This will re-trigger this function, so we return early here.
        server.state.color_range = [min_range, max_range]
        return
        
    actor1.SetVisibility(show_data)
    actor2.SetVisibility(show_data)

    # 1. Update vtkLookupTable
    lut.SetTableRange(min_range, max_range)
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.6667, 0.0)
    
    lut.SetUseAboveRangeColor(use_above)
    if use_above:
        lut.SetAboveRangeColor(*hex_to_rgba(above_color))
        
    lut.SetUseBelowRangeColor(use_below)
    if use_below:
        lut.SetBelowRangeColor(*hex_to_rgba(below_color))
        
    lut.SetNanColor(*hex_to_rgba(nan_color))
    
    if use_log:
        lut.SetScaleToLog10()
    else:
        lut.SetScaleToLinear()
        
    lut.Build()

    # 2. Update vtkColorTransferFunction
    ctf.RemoveAllPoints()
    ctf.SetColorSpaceToRGB() # Use RGB interpolation between the 256 points
    
    if use_log:
        ctf.SetScaleToLog10()
    else:
        ctf.SetScaleToLinear()
        
    # Copy the 256 exact colors from the vtkLookupTable to ensure they are identical
    import math
    
    lut.ResetAnnotations()
    ctf.ResetAnnotations()
    
    # 3. Add explicit annotations for exactly 5 nodes to ensure both scalar bars match perfectly
    for i in range(5):
        if use_log and min_range > 0 and max_range > 0:
            val = 10 ** (math.log10(min_range) + (math.log10(max_range) - math.log10(min_range)) * (i / 4.0))
        else:
            val = min_range + (max_range - min_range) * (i / 4.0)
            
        label = f"{val:.2f}"
        lut.SetAnnotation(val, label)
        ctf.SetAnnotation(val, label)

    for i in range(256):
        if use_log and min_range > 0 and max_range > 0:
            # Sample at the exact logarithmic bin boundaries that vtkLookupTable uses
            log_min = math.log10(min_range)
            log_max = math.log10(max_range)
            val = 10 ** (log_min + (log_max - log_min) * (i / 255.0))
        else:
            val = min_range + (max_range - min_range) * (i / 255.0)
            
        color = [0.0, 0.0, 0.0]
        lut.GetColor(val, color)
        ctf.AddRGBPoint(val, color[0], color[1], color[2])
    
    ctf.SetUseAboveRangeColor(use_above)
    if use_above:
        ctf.SetAboveRangeColor(*hex_to_rgb(above_color))
        
    ctf.SetUseBelowRangeColor(use_below)
    if use_below:
        ctf.SetBelowRangeColor(*hex_to_rgb(below_color))
        
    ctf.SetNanColor(*hex_to_rgb(nan_color))
    
    ctf.Build()
    mapper1.Modified()
    mapper2.Modified()
    
    # Render both windows
    renderWindow1.Render()
    renderWindow2.Render()
    if hasattr(ctrl, "view_update_1"):
        ctrl.view_update_1()
    if hasattr(ctrl, "view_update_2"):
        ctrl.view_update_2()

# Initial defaults
state.show_data = True
state.data_range = [-10, 10]
state.color_range = [-5, 5]
state.use_log = False
state.use_above = True
state.above_color = "#00FF00"
state.use_below = True
state.below_color = "#FFFF00"
state.nan_color = "#FF00FF"

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Color Map Comparison")
    
    with layout.drawer:
        with vuetify.VContainer():
            vuetify.VSwitch(v_model=("show_data",), label="Show Data", color="primary")
            vuetify.VSwitch(v_model=("use_log",), label="Log Scale", color="primary")
            
            vuetify.VRangeSlider(
                v_model=("data_range",),
                min=-500, max=500, step=1,
                label="Data Bounds",
                thumb_label="always",
                class_="mt-6"
            )
            
            vuetify.VRangeSlider(
                v_model=("color_range",),
                min=-500, max=500, step=0.1,
                label="Mapping Bounds",
                thumb_label="always",
                class_="mt-6"
            )
            
            vuetify.VSwitch(v_model=("use_above",), label="Above Range Color", color="primary")
            vuetify.VColorPicker(v_model=("above_color",), v_show="use_above", hide_inputs=True, width="100%", elevation=0)
            
            vuetify.VSwitch(v_model=("use_below",), label="Below Range Color", color="primary")
            vuetify.VColorPicker(v_model=("below_color",), v_show="use_below", hide_inputs=True, width="100%", elevation=0)
            
            vuetify.VLabel("NaN Color")
            vuetify.VColorPicker(v_model=("nan_color",), hide_inputs=True, width="100%", elevation=0)
        
    with layout.content:
        with vuetify.VRow(classes="ma-0 fill-height"):
            with vuetify.VCol(cols=6, classes="pa-0"):
                view1 = vtk_widgets.VtkRemoteView(renderWindow1, ref="view1")
                ctrl.view_update_1 = view1.update
            with vuetify.VCol(cols=6, classes="pa-0"):
                view2 = vtk_widgets.VtkRemoteView(renderWindow2, ref="view2")
                ctrl.view_update_2 = view2.update

if __name__ == "__main__":
    server.start()
