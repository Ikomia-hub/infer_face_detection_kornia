# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from distutils.util import strtobool
import torch
import kornia as K
import copy
from kornia.contrib import FaceDetector, FaceDetectorResult


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFaceDetectionKorniaParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.face_detection = None
        self.cuda = torch.cuda.is_available()
        self.conf_thres = 0.6
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        self.cuda = strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["conf_thres"] = str(self.conf_thres)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFaceDetectionKornia(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create parameters class
        if param is None:
            self.setParam(InferFaceDetectionKorniaParam())
        else:
            self.setParam(copy.deepcopy(param))
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        self.face_detection = None

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def predict(self, src_image):
        # Call to the process main routine
        # Preprocess
        proc_img  = K.image_to_tensor(src_image, keepdim=False).to(self.device, torch.float32)

        # Detect faces
        with torch.no_grad():
            dets = self.face_detection(proc_img)

        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("kornia")
        graphics_output.setImageIndex(0)

        # to decode later the detections
        dets = [FaceDetectorResult(o) for o in dets]

        param = self.getParam()

        for b in dets:  
            if b.score < param.conf_thres:  # skip detections with lower score
                continue
            # draw face bounding box around each detected face
            x1, y1 = b.top_left.int().tolist()
            x2, y2 = b.bottom_right.int().tolist()
            w = float(x2 - x1)
            h = float(y2 - y1)
            prop_rect = core.GraphicsRectProperty()
            graphics_box = graphics_output.addRectangle(x1, y1, w, h, prop_rect)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get parameters :
        param = self.getParam()
        if param.update or self.face_detection is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            # Create the detector
            self.face_detection = FaceDetector().to(self.device, torch.float32)
            param.update = False
            print("Will run on {}".format(self.device.type))

        src_image = input.getImage()

        self.predict(src_image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFaceDetectionKorniaFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_face_detection_kornia"
        self.info.shortDescription = "Face detection using the Kornia API"
        self.info.description = "This plugin propose inference for multi-face detection"\
                                "using Kornia based one the YuNet model." \
                                "The model implementation is based on Pytorch framework." \
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski"
        self.info.article = "Kornia: an Open Source Differentiable"\
                            "Computer Vision Library for PyTorch"
        self.info.journal = "https://arxiv.org/pdf/1910.02190.pdf"
        self.info.year = 2020
        self.info.license = "Apache-2.0"
        # URL of documentation
        self.info.documentationLink = "https://kornia.readthedocs.io/en/latest/applications/"\
                                    "face_detection.html"
        # Code source repository
        self.info.repository = "https://github.com/kornia/kornia/tree/master/examples/face_detection"
        # Keywords used for search
        self.info.keywords = "face detection, kornia, Yunet, cv2, Pytorch "

    def create(self, param=None):
        # Create process object
        return InferFaceDetectionKornia(self.info.name, param)
