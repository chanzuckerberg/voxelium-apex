#!/usr/bin/env python

"""
File for testing VTK
"""


from voxelium_apex.sha3d.vtk_utils import initialize_vtk_resourses, make_cylinder_actor


if __name__ == "__main__":
    render_window, renderer, interactor = initialize_vtk_resourses()
    
    actor = make_cylinder_actor()
    renderer.AddActor(actor)

    interactor.Initialize()

    render_window.Render()
    interactor.Start()
    interactor.TerminateApp()

    
