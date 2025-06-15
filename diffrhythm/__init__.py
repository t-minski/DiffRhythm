import onnxruntime as ort
ort.set_default_logger_severity(4)

import warnings
warnings.filterwarnings("ignore", message="Full dictionary is not installed for 'zh'")
