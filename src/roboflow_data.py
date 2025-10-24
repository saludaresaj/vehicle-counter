from roboflow import Roboflow
rf = Roboflow(api_key="s7OTYIyk08jMGdYmBtts")
project = rf.workspace("kt-x1due").project("miniproject4_1")
version = project.version(11)
dataset = version.download("yolov8")