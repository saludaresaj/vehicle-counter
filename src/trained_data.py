from roboflow import Roboflow
rf = Roboflow(api_key="s7OTYIyk08jMGdYmBtts")
project = rf.workspace("my-workspace-sw2sq").project("vehicle-counting-ha3si")
version = project.version(2)
dataset = version.download("yolov8")
                