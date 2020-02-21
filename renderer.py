
import numpy as np
import OpenGL.GL as gl
import pangolin as pango

class Renderer():

    def __init__(self, name):
        super().__init__()

        self.name = name

        # Window
        pango.CreateWindowAndBind(self.name, int(640*(3/2)), int(480*(3/2)))
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pango.ModelViewLookAt(0, -0.5, -0.5, 0, 0, 1, 0, -1, 0))
        self.handler = pango.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pango.CreateDisplay()
        self.dcam.SetBounds(
            pango.Attach(0), 
            pango.Attach(1), 
            pango.Attach(0), 
            pango.Attach(1), -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def should_quit(self):
        return pango.ShouldQuit()

    def clear_screen(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def draw_points(self, points, colors):
        self.dcam.Activate(self.scam)
        gl.glPointSize(5)
        pango.DrawPoints(points, colors)
    
    def draw_camera(self):
        self.dcam.Activate(self.scam)
        pose = np.identity(4)
        pose[:3, 3] = 0
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pango.DrawCamera(pose, 0.5, 0.75, 0.8)

    def finish_frame(self):
        pango.FinishFrame()
