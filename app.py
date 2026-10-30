import math
import numpy as np
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify, vtk as vtk_widgets
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# -----------------------------------------------------------------------------
# VTK Pipeline
# -----------------------------------------------------------------------------
source = vtk.vtkRTAnalyticSource()
source.SetWholeExtent(-10, 10, -10, 10, -10, 10)
source.Update()

dataset = vtk.vtkImageData()
dataset.DeepCopy(source.GetOutput())

scalars = dataset.GetPointData().GetScalars()
arr = vtk_to_numpy(scalars).astype(np.float64)

original_arr = vtk_to_numpy(scalars).astype(np.float64).copy()
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
    hex_color = hex_color.lstrip("#")
    if len(hex_color) >= 6:
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (1.0, 1.0, 1.0)


def hex_to_rgba(hex_color):
    rgb = hex_to_rgb(hex_color)
    return rgb + (1.0,)


def symlog(x, lt):
    return math.copysign(math.log10(1.0 + abs(x) / lt), x)


def symlog_inv(y, lt):
    return math.copysign(lt * (10.0 ** abs(y) - 1.0), y)


def format_val(val):
    if abs(val) == 0:
        return "0.0"
    if abs(val) < 1e-2 or abs(val) >= 1e4:
        return f"{val:.1e}"
    return f"{val:.2f}"


PRESETS = {
    "Default ([-10, 10])": {
        "data_range": [-10.0, 10.0],
        "color_range": [-5.0, 5.0],
        "scale_mode": "Linear",
        "linthresh": 10.0,
    },
    "QuickView HDR: ±1e6 (linthresh 1e-7)": {
        "data_range": [-1e6, 1e6],
        "color_range": [-1e6, 1e6],
        "scale_mode": "Symlog",
        "linthresh": 1e-7,
    },
    "QuickView #114 Trace: ±5.56e-10 (linthresh 1e-11)": {
        "data_range": [-5.56e-10, 5.56e-10],
        "color_range": [-5.56e-10, 5.56e-10],
        "scale_mode": "Symlog",
        "linthresh": 1e-11,
    },
    "QuickView SHFLX: [-424, 418] (linthresh 5.0)": {
        "data_range": [-424.0, 418.0],
        "color_range": [-424.0, 418.0],
        "scale_mode": "Symlog",
        "linthresh": 5.0,
    },
}


@state.change("data_range")
def update_data_array(data_range, **kwargs):
    d_min, d_max = float(data_range[0]), float(data_range[1])
    span = abs(d_max - d_min)
    pad = span * 0.05 if span > 0 else 1.0

    # Scale array to [d_min, d_max]
    arr = (original_arr - original_min) / (original_max - original_min) * (
        d_max - d_min
    ) + d_min

    # Reshape and apply out of bounds/NaN chunks
    arr_3d = arr.reshape((21, 21, 21))
    arr_3d[10:21, 10:21, 10:21] = np.nan
    arr_3d[10:21, 0:10, 10:21] = d_min - pad
    arr_3d[0:10, 10:21, 10:21] = d_max + pad

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


@state.change("selected_preset")
def apply_preset(selected_preset, **kwargs):
    if selected_preset in PRESETS:
        p = PRESETS[selected_preset]
        state.scale_mode = p["scale_mode"]
        state.linthresh = float(p["linthresh"])
        state.data_range = [float(p["data_range"][0]), float(p["data_range"][1])]
        state.color_range = [float(p["color_range"][0]), float(p["color_range"][1])]
        state.data_min_str = (
            f"{p['data_range'][0]:.2e}"
            if abs(p["data_range"][0]) < 1e-2 or abs(p["data_range"][0]) >= 1e4
            else str(p["data_range"][0])
        )
        state.data_max_str = (
            f"{p['data_range'][1]:.2e}"
            if abs(p["data_range"][1]) < 1e-2 or abs(p["data_range"][1]) >= 1e4
            else str(p["data_range"][1])
        )
        state.color_min_str = (
            f"{p['color_range'][0]:.2e}"
            if abs(p["color_range"][0]) < 1e-2 or abs(p["color_range"][0]) >= 1e4
            else str(p["color_range"][0])
        )
        state.color_max_str = (
            f"{p['color_range'][1]:.2e}"
            if abs(p["color_range"][1]) < 1e-2 or abs(p["color_range"][1]) >= 1e4
            else str(p["color_range"][1])
        )
        state.linthresh_str = (
            f"{p['linthresh']:.2e}"
            if p["linthresh"] < 1e-2 or p["linthresh"] >= 1e4
            else str(p["linthresh"])
        )


def parse_and_apply_inputs():
    try:
        d_min = float(state.data_min_str)
        d_max = float(state.data_max_str)
        if d_min < d_max:
            state.data_range = [d_min, d_max]
    except ValueError, TypeError:
        pass

    try:
        c_min = float(state.color_min_str)
        c_max = float(state.color_max_str)
        if c_min < c_max:
            state.color_range = [c_min, c_max]
    except ValueError, TypeError:
        pass

    try:
        lt = float(state.linthresh_str)
        if lt > 0:
            state.linthresh = lt
    except ValueError, TypeError:
        pass


@state.change(
    "use_above",
    "above_color",
    "use_below",
    "below_color",
    "nan_color",
    "scale_mode",
    "linthresh",
    "color_range",
    "show_data",
)
def update_colormap(
    use_above,
    above_color,
    use_below,
    below_color,
    nan_color,
    scale_mode,
    linthresh,
    color_range,
    show_data,
    **kwargs,
):
    min_range, max_range = float(color_range[0]), float(color_range[1])
    linthresh = float(linthresh)

    # Implement ParaView's AdjustRangeForLog logic
    if scale_mode == "Log" and (min_range <= 0 or max_range <= 0):
        if max_range <= 0:
            min_range = 1.0e-4
            max_range = 1.0
        else:
            min_range = max_range * 0.0001
            min_range = min_range if min_range < 1.0 else 1.0

        server.state.color_range = [min_range, max_range]
        return

    actor1.SetVisibility(show_data)
    actor2.SetVisibility(show_data)

    # 1. Update vtkLookupTable
    lut.SetScaleToLinear()
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

    if scale_mode == "Log":
        lut.SetScaleToLog10()
    elif scale_mode == "Symlog":
        lut.SetScaleToSymlog()
        lut.SetSymlogLinearThreshold(linthresh)

    lut.Build()

    # 2. Update vtkColorTransferFunction
    ctf.RemoveAllPoints()
    ctf.SetColorSpaceToRGB()

    if scale_mode == "Log":
        ctf.SetScaleToLog10()
    elif scale_mode == "Symlog":
        ctf.SetScaleToSymlog()
        ctf.SetSymlogLinearThreshold(linthresh)
    else:
        ctf.SetScaleToLinear()

    lut.ResetAnnotations()
    ctf.ResetAnnotations()

    # 3. Add explicit annotations for 5 nodes
    for i in range(5):
        if scale_mode == "Log" and min_range > 0 and max_range > 0:
            val = 10 ** (
                math.log10(min_range)
                + (math.log10(max_range) - math.log10(min_range)) * (i / 4.0)
            )
        elif scale_mode == "Symlog":
            s_min = symlog(min_range, linthresh)
            s_max = symlog(max_range, linthresh)
            val = symlog_inv(s_min + (s_max - s_min) * (i / 4.0), linthresh)
            val = max(min_range, min(max_range, val))
        else:
            val = min_range + (max_range - min_range) * (i / 4.0)

        label = format_val(val)
        lut.SetAnnotation(val, label)
        ctf.SetAnnotation(val, label)

    # Sample 256 exact points
    for i in range(256):
        if scale_mode == "Log" and min_range > 0 and max_range > 0:
            log_min = math.log10(min_range)
            log_max = math.log10(max_range)
            val = 10 ** (log_min + (log_max - log_min) * (i / 255.0))
        elif scale_mode == "Symlog":
            s_min = symlog(min_range, linthresh)
            s_max = symlog(max_range, linthresh)
            val = symlog_inv(s_min + (s_max - s_min) * (i / 255.0), linthresh)
            val = max(min_range, min(max_range, val))
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
state.data_range = [-10.0, 10.0]
state.color_range = [-5.0, 5.0]
state.scale_mode = "Linear"
state.linthresh = 10.0
state.use_above = True
state.above_color = "#00FF00"
state.use_below = True
state.below_color = "#FFFF00"
state.nan_color = "#FF00FF"
state.selected_preset = "Default ([-10, 10])"
state.data_min_str = "-10.0"
state.data_max_str = "10.0"
state.color_min_str = "-5.0"
state.color_max_str = "5.0"
state.linthresh_str = "10.0"

ctrl.parse_and_apply_inputs = parse_and_apply_inputs

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Color Map Comparison")

    with layout.drawer:
        with vuetify.VContainer():

            vuetify.VSwitch(v_model=("show_data",), label="Show Data", color="primary")
            vuetify.VSelect(
                v_model=("selected_preset", "Default ([-10, 10])"),
                items=("presets", list(PRESETS.keys())),
                label="QuickView Range Presets",
                color="primary",
            )
            vuetify.VSelect(
                v_model=("scale_mode", "Linear"),
                items=("scale_modes", ["Linear", "Log", "Symlog"]),
                label="Scale",
                color="primary",
            )
            vuetify.VTextField(
                v_model=("linthresh_str", "10.0"),
                label="Linthresh (Symlog)",
                v_if="scale_mode === 'Symlog'",
                change=ctrl.parse_and_apply_inputs,
                density="compact",
            )

            # Manual string inputs for high dynamic range & trace concentrations
            with vuetify.VRow(classes="mt-1"):
                with vuetify.VCol(cols=6):
                    vuetify.VTextField(
                        v_model=("data_min_str", "-10.0"),
                        label="Data Min",
                        density="compact",
                        change=ctrl.parse_and_apply_inputs,
                    )
                with vuetify.VCol(cols=6):
                    vuetify.VTextField(
                        v_model=("data_max_str", "10.0"),
                        label="Data Max",
                        density="compact",
                        change=ctrl.parse_and_apply_inputs,
                    )

            with vuetify.VRow(classes="mt-1"):
                with vuetify.VCol(cols=6):
                    vuetify.VTextField(
                        v_model=("color_min_str", "-5.0"),
                        label="Map Min",
                        density="compact",
                        change=ctrl.parse_and_apply_inputs,
                    )
                with vuetify.VCol(cols=6):
                    vuetify.VTextField(
                        v_model=("color_max_str", "5.0"),
                        label="Map Max",
                        density="compact",
                        change=ctrl.parse_and_apply_inputs,
                    )

            vuetify.VSwitch(
                v_model=("use_above",),
                label="Above Range Color",
                color="primary",
            )
            vuetify.VColorPicker(
                v_model=("above_color",),
                v_show="use_above",
                hide_inputs=True,
                width="100%",
                elevation=0,
            )

            vuetify.VSwitch(
                v_model=("use_below",),
                label="Below Range Color",
                color="primary",
            )
            vuetify.VColorPicker(
                v_model=("below_color",),
                v_show="use_below",
                hide_inputs=True,
                width="100%",
                elevation=0,
            )

            vuetify.VLabel("NaN Color")
            vuetify.VColorPicker(
                v_model=("nan_color",),
                hide_inputs=True,
                width="100%",
                elevation=0,
            )

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
