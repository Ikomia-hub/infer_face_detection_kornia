# PyQt GUI framework
from PyQt6.QtWidgets import *

from torch.cuda import is_available

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_face_detection_kornia.infer_face_detection_kornia_process import InferFaceDetectionKorniaParam


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferFaceDetectionKorniaWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferFaceDetectionKorniaParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Threshold
        self.double_spin_thres = pyqtutils.append_double_spin(
                                self.gridLayout, "Confidence threshold",
                                self.parameters.conf_thres, min = 0., max = 1., step = 1e-1)

        # CUDA
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda", self.parameters.cuda and is_available())

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Get parameters from widget
        self.parameters.conf_thres = self.double_spin_thres.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferFaceDetectionKorniaWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process ->
        # it must be the same as the one declared in the process factory class
        self.name = "infer_face_detection_kornia"

    def create(self, param):
        # Create widget object
        return InferFaceDetectionKorniaWidget(param, None)
