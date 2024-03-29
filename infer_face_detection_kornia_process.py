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
from ikomia.utils import strtobool
import torch
import kornia as K
import copy
from kornia.contrib import FaceDetector, FaceDetectorResult
import cv2
import os

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFaceDetectionKorniaParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.conf_thres = 0.6
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        self.cuda = strtobool(params["cuda"])
        self.conf_thres = float(params["conf_thres"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {"cuda": str(self.cuda),
                  "conf_thres": str(self.conf_thres)}
        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFaceDetectionKornia(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create parameters class
        if param is None:
            self.set_param_object(InferFaceDetectionKorniaParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.face_detection = None
        self.names = ["face"]
        self.max_size = 1000000
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def resize_image(self, img, width, height):
        # Check if the image width or height is greater than 1000
        if width > 1000 or height > 1000:
            # Calculate the aspect ratio
            aspect_ratio = width / height

            # If width is greater than height
            if aspect_ratio > 1:
                new_width = 1000
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 1000
                new_width = int(new_height * aspect_ratio)

            # Resize the image
            img = cv2.resize(img, (new_width, new_height))

        return img, new_height, new_width

    def predict(self, src_image, h_ratio, w_ratio):
        # Call to the process main routine
        # Preprocess
        proc_img  = K.image_to_tensor(src_image, keepdim=False).to(self.device, torch.float32)

        # Detect faces
        with torch.no_grad():
            dets = self.face_detection(proc_img)

        # to decode later the detections
        dets = [FaceDetectorResult(o) for o in dets]

        param = self.get_param_object()

        self.set_names(self.names)
        for i, b in enumerate(dets):
            if b.score < param.conf_thres: # skip detections with lower score
                continue
            # draw face bounding box around each detected face
            x1, y1 = b.top_left.int().tolist()
            x2, y2 = b.bottom_right.int().tolist()
            x1, y1 = x1 * w_ratio, y1 * h_ratio
            x2, y2 = x2 * w_ratio, y2 * h_ratio
            w = float(x2 - x1)
            h = float(y2 - y1)
            self.add_object(i+1, 0, b.score.item(), float(x1), float(y1), w, h)

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Temporary fix to clean detection outputs at the beginning of the task
        self.get_output(1).clear_data()

        # Get input :
        input = self.get_input(0)

        # Get parameters :
        param = self.get_param_object()
        if param.update or self.face_detection is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            # Create the detector     
            if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)
            torch_dir_ori = torch.hub.get_dir()
            torch.hub.set_dir(self.model_folder)
            self.face_detection = FaceDetector().to(self.device, torch.float32)
            param.update = False
            print("Will run on {}".format(self.device.type))
            torch.hub.set_dir(torch_dir_ori)

        src_image = input.get_image()

        # Face detection losses accuracy on large images, images are resized to training size. 
        height, width = src_image.shape[:2]

        if height * width > self.max_size:
            src_image, new_height, new_width = self.resize_image(src_image, width, height)
            height_ratio, width_ratio = height / new_height, width / new_width
            self.predict(src_image, height_ratio, width_ratio)
        else:
            self.predict(src_image, 1, 1)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFaceDetectionKorniaFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_face_detection_kornia"
        self.info.short_description = "Face detection using the Kornia API"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.2"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski"
        self.info.article = "Kornia: an Open Source Differentiable"\
                            "Computer Vision Library for PyTorch"
        self.info.journal = "https://arxiv.org/pdf/1910.02190.pdf"
        self.info.year = 2020
        self.info.license = "Apache-2.0"
        # URL of documentation
        self.info.documentation_link = "https://kornia.readthedocs.io/en/latest/applications/"\
                                    "face_detection.html"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_face_detection_kornia"
        self.info.original_repository = "https://github.com/kornia/kornia/tree/master/examples/face_detection"
        # Keywords used for search
        self.info.keywords = "face detection, kornia, Yunet, cv2, Pytorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create process object
        return InferFaceDetectionKornia(self.info.name, param)
