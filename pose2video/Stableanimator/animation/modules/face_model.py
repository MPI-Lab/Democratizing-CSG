from torch import nn
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import insightface
from insightface.app import FaceAnalysis


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=-1)

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device="cpu",
        )
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device="cpu")
