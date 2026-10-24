import numpy as np
import vtk
from vtk.util import numpy_support

def vtk_visualize_cxd(imgs, com, combined_results, peaks, B_recip, thickness=1.0, save_path="vtk_output.png", show=True):
    """
    Render the volume and the paired symmetry axes in reciprocal space with VTK.
    White background, blue particle; saves a PNG to `save_path` when set.
    """
    # float32 because boolean images may not render
    imgs_float = imgs.astype(np.float32) 
    
    renderer = vtk.vtkRenderer()
    render_win = vtk.vtkRenderWindow()
    render_win.AddRenderer(renderer)
    render_win.SetSize(1000, 800)
    
    if not show:
        render_win.SetOffScreenRendering(1) 

    # B_recip is in m^-1; scale it down for display
    vis_scale = 3 * 1e-6 
    
    # q = B_recip @ (n - com)
    transform = vtk.vtkTransform()
    transform.Scale(vis_scale, vis_scale, vis_scale) 
    
    vtk_mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            vtk_mat.SetElement(i, j, B_recip[i, j])
    transform.Concatenate(vtk_mat)
    transform.Translate(-com[2], -com[1], -com[0])  # com is (z, y, x); VTK expects (x, y, z)

    vtk_data = numpy_support.numpy_to_vtk(num_array=imgs_float.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    volume_data = vtk.vtkImageData()
    volume_data.SetDimensions(imgs.shape[2], imgs.shape[1], imgs.shape[0])
    volume_data.GetPointData().SetScalars(vtk_data)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(0.5, 0.2)
    opacity_func.AddPoint(1.0, 0.7)
    volume_property.SetScalarOpacity(opacity_func)

    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)  # same as the background
    color_func.AddRGBPoint(1.0, 0.1, 0.4, 0.8)
    volume_property.SetColor(color_func)

    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(volume_data)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_property)
    volume_actor.SetUserTransform(transform) 
    renderer.AddActor(volume_actor)

    line_len = np.linalg.norm(B_recip @ np.array(imgs.shape)) * 0.4 * vis_scale
    for i, res in enumerate(combined_results):
        p_idx = res["pair"][0]
        dir_idx = np.array([peaks[p_idx][2], peaks[p_idx][1], peaks[p_idx][0]])
        dir_q = B_recip @ dir_idx
        dir_q /= np.linalg.norm(dir_q)

        line = vtk.vtkLineSource()
        line.SetPoint1(-(dir_q * line_len))
        line.SetPoint2(dir_q * line_len)
        
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(line.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetLineWidth(3 * thickness)
        # dark gray to contrast with the white background
        a.GetProperty().SetColor(0.2, 0.2, 0.2) 
        renderer.AddActor(a)

    axes_actor = vtk.vtkAxesActor()
    axes_actor.SetXAxisLabelText("Qx")
    axes_actor.SetYAxisLabelText("Qy")
    axes_actor.SetZAxisLabelText("Qz")
    # black labels for the white background
    axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)
    axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)
    axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_win)

    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes_actor)
    marker.SetInteractor(interactor)
    marker.SetViewport(0.0, 0.0, 0.2, 0.2)
    marker.SetEnabled(1)
    marker.InteractiveOff()

    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.ResetCamera()
    
    # Zoom must come after ResetCamera(), which resets it
    renderer.GetActiveCamera().Zoom(2.5) 

    render_win.Render()

    if save_path:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_win)
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(save_path))
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        print(f"이미지가 저장되었습니다: {save_path}")

    if show:
        interactor.Start()
    
    render_win.Finalize()
    del render_win